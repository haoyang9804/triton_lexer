from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle

from fla.ops.utils import logsumexp_fwd
from fla.ops.utils.op import exp
from fla.utils import input_guard


MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def cross_entropy_kernel(
    logits,
    lse,
    target,
    loss,
    total,
    ignore_index,
    label_smoothing: tl.constexpr,
    logit_scale: tl.constexpr,
    reduction: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):

    i_n = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)

    b_y = tl.load(target + i_n)

    logits += i_n * V

    if b_y == ignore_index:

        for i in range(0, V, BV):
            o_v = i + tl.arange(0, BV)
            tl.store(logits + o_v, 0.0, mask=o_v < V)
        return

    b_l = tl.load(logits + b_y) * logit_scale
    b_lse = tl.load(lse + i_n)

    b_loss = b_lse - b_l

    b_z = 0.0
    eps = label_smoothing / V

    tl.debug_barrier()

    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        b_logits = (
            tl.load(logits + o_v, mask=o_v < V, other=float("-inf")) * logit_scale
        )
        if label_smoothing > 0:

            b_z += tl.sum(tl.where(o_v < V, -eps * b_logits, 0.0))
        b_p = (exp(b_logits - b_lse) - eps) * logit_scale
        if reduction == "mean":
            b_p = b_p / total
        tl.store(logits + o_v, b_p, mask=o_v < V)

        tl.debug_barrier()

    if label_smoothing > 0:
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing * b_lse)

    b_l = tl.load(logits + b_y)

    if reduction == "mean":
        b_loss = b_loss / total
        b_l += (label_smoothing - 1) / total * logit_scale
    else:
        b_l += (label_smoothing - 1) * logit_scale

    tl.store(loss + i_n, b_loss)
    tl.store(logits + b_y, b_l)


@triton.jit
def elementwise_mul_kernel(x, g, N: tl.constexpr, B: tl.constexpr):

    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)

    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def fused_linear_cross_entropy_forward(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
    reduction: str = "mean",
):
    device = x.device

    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    dx = torch.zeros_like(x, device=device)

    dw = (
        torch.zeros_like(weight, device=device, dtype=torch.float)
        if weight is not None
        else None
    )

    db = (
        torch.zeros_like(bias, device=device, dtype=torch.float)
        if bias is not None
        else None
    )

    loss = torch.zeros(N, device=device, dtype=torch.float)

    total = target.ne(ignore_index).sum().item()

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)

        c_x = x[start:end]

        c_logits = F.linear(c_x, weight, bias)
        c_target = target[start:end]

        c_lse = logsumexp_fwd(c_logits, scale=logit_scale, dtype=torch.float)

        c_loss = loss[start:end]

        cross_entropy_kernel[(c_logits.shape[0],)](
            logits=c_logits,
            lse=c_lse,
            target=c_target,
            loss=c_loss,
            total=total,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            reduction=reduction,
            V=V,
            BV=BV,
            num_warps=32,
        )

        dx[start:end] = torch.mm(c_logits, weight)

        if weight is not None:
            dw += c_logits.t() @ c_x

        if bias is not None:
            torch.add(input=db, other=c_logits.sum(0), out=db)

    loss = loss.sum()
    if dw is not None:
        dw = dw.to(weight)
    if db is not None:
        db = db.to(bias)
    return loss, dx, dw, db


def fused_linear_cross_entropy_backward(
    do: torch.Tensor, dx: torch.Tensor, dw: torch.Tensor, db: torch.Tensor
):

    if torch.ne(do, torch.tensor(1.0, device=do.device)):

        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
            x=dx,
            g=do,
            N=N * H,
            B=B,
            num_warps=32,
        )

        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[(triton.cdiv(V * H, B),)](
                x=dw,
                g=do,
                N=V * H,
                B=B,
                num_warps=32,
            )

        if db is not None:
            V = db.shape[0]
            elementwise_mul_kernel[(triton.cdiv(V, B),)](
                x=db,
                g=do,
                N=V,
                B=B,
                num_warps=32,
            )
    return dx, dw, db


class FusedLinearCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        num_chunks: int = 8,
        reduction: str = "mean",
    ):

        loss, dx, dw, db = fused_linear_cross_entropy_forward(
            x,
            target,
            weight,
            bias,
            ignore_index,
            label_smoothing,
            logit_scale,
            num_chunks,
            reduction,
        )

        ctx.save_for_backward(
            dx.detach(),
            dw.detach() if weight is not None else None,
            db.detach() if bias is not None else None,
        )
        return loss

    @staticmethod
    @input_guard
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        dx, dw, db = fused_linear_cross_entropy_backward(do, dx, dw, db)
        return dx, None, dw, db, None, None, None, None, None


def fused_linear_cross_entropy_loss(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:

    return FusedLinearCrossEntropyFunction.apply(
        x,
        target,
        weight,
        bias,
        ignore_index,
        label_smoothing,
        logit_scale,
        num_chunks,
        reduction,
    )


class FusedLinearCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        num_chunks: int = 8,
        reduction: str = "mean",
    ):

        super().__init__()

        assert reduction in ["mean", "sum"], f"reduction: {reduction} is not supported"

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.num_chunks = num_chunks
        self.reduction = reduction

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):

        loss = fused_linear_cross_entropy_loss(
            x.view(-1, x.shape[-1]),
            target.view(-1),
            weight=weight,
            bias=bias,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            num_chunks=self.num_chunks,
            reduction=self.reduction,
        )
        return loss


class LinearLossParallel(ParallelStyle):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        use_local_output: bool = False,
    ):
        super().__init__()

        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        x, target, weight, bias = inputs

        if not isinstance(x, DTensor):

            x = DTensor.from_local(x, device_mesh, sequence_sharding)
        if x.placements != sequence_sharding:
            x = x.redistribute(placements=sequence_sharding, async_op=True)
        if not isinstance(target, DTensor):
            target = DTensor.from_local(target, device_mesh, [Replicate()])
        if target.placements != sequence_sharding:
            target = target.redistribute(placements=sequence_sharding, async_op=True)

        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, [Replicate()])
        if weight.placements != [Replicate()]:

            weight = weight.redistribute(placements=[Replicate()], async_op=True)

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, [Replicate()])
        if bias is not None and bias.placements != [Replicate()]:
            bias = bias.redistribute(placements=[Replicate()], async_op=True)

        return (
            x.to_local(),
            target.to_local(),
            weight.to_local(),
            bias.to_local() if bias is not None else bias,
        )

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=partial(self._prepare_input_fn, self.sequence_sharding),
            output_fn=partial(self._prepare_output_fn, self.use_local_output),
        )
