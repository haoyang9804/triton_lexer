from typing import Literal

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device


def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


MAX_FUSED_SIZE = 65536 // 4

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]

_REDUCTION_MODE_NONE: tl.constexpr = tl.constexpr(0)
_REDUCTION_MODE_SUM: tl.constexpr = tl.constexpr(1)
_REDUCTION_MODE_MEAN: tl.constexpr = tl.constexpr(2)
_REDUCTION_MODE_BATCHMEAN: tl.constexpr = tl.constexpr(3)

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE.value,
    "sum": _REDUCTION_MODE_SUM.value,
    "mean": _REDUCTION_MODE_MEAN.value,
    "batchmean": _REDUCTION_MODE_BATCHMEAN.value,
}


@triton.jit
def _kldiv_kernel_forward(
    y_ptr,
    y_stride,
    gt_ptr,
    gt_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)


@triton.jit
def _kldiv_kernel_backward(
    target_ptr,
    target_stride,
    new_grads_ptr,
    new_grads_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)

    target_ptr += pid * target_stride
    new_grads_ptr += pid * new_grads_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)

        tl.store(new_grads_ptr + offsets, res, mask=mask)


def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):
    BT, V = y_pred.shape
    BLOCK_SIZE = (
        min(8192, triton.next_power_of_2(V))
        if infer_device() == "xpu"
        else min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    )
    num_warps = 32 if infer_device() == "xpu" else get_num_warps(BLOCK_SIZE)

    grid = (BT,)
    reduction = _str_to_reduction_mode[reduction]

    out_size = (BT, V) if reduction == _REDUCTION_MODE_NONE.value else (BT,)
    output_tensor = torch.zeros(out_size, device=y_pred.device, dtype=torch.float32)

    _kldiv_kernel_forward[grid](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        output_tensor,
        output_tensor.stride(0),
        V,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
        reduction=reduction,
    )

    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return output_tensor.sum() / BT
    elif reduction == _REDUCTION_MODE_SUM.value:
        return output_tensor.sum(dim=0)
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return output_tensor.sum() / (BT * V)
    else:
        return output_tensor


def kldiv_backward_triton(target, grad_output, new_grads, log_target):
    BT, V = target.shape
    BLOCK_SIZE = (
        min(8192, triton.next_power_of_2(V))
        if infer_device() == "xpu"
        else min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    )
    num_warps = 32 if infer_device() == "xpu" else get_num_warps(BLOCK_SIZE)

    grid = (BT,)

    _kldiv_kernel_backward[grid](
        target,
        target.stride(0),
        new_grads,
        new_grads.stride(0),
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
    )

    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return new_grads

    return new_grads * grad_output


class LigerKLDivLossFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: REDUCTION_LITERAL = "batchmean",
        log_target: bool = False,
        eps: float = 1e-10,
    ) -> torch.Tensor:

        ctx.save_for_backward(y_true)
        ctx.reduction = reduction
        ctx.log_target = log_target
        return kldiv_forward_triton(
            y_pred, y_true, log_target=log_target, reduction=reduction, eps=eps
        )

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:

        (y_true,) = ctx.saved_tensors

        new_grads = torch.empty_like(y_true)

        derivative = kldiv_backward_triton(
            y_true, grad_output, new_grads, ctx.log_target
        )

        if ctx.reduction == "batchmean":
            derivative = derivative / y_true.shape[0]
        elif ctx.reduction == "sum" or ctx.reduction == "none":
            pass
        elif ctx.reduction == "mean":
            derivative = derivative / (y_true.shape[0] * y_true.shape[1])

        return (
            derivative,
            None,
            None,
            None,
            None,
        )
