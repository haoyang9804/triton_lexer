import torch
import triton

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip


MAX_FUSED_SIZE = 65536 // 2


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
):
    assert isinstance(
        return_z_loss, bool
    ), f"return_z_loss must be True or False. Got: {return_z_loss}"
    device = _input.device

    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    grad_weight = (
        torch.zeros_like(weight, device=device) if weight.requires_grad else None
    )
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = (
        torch.zeros(BT, dtype=_input.dtype, device=_input.device)
        if return_z_loss
        else None
    )

    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert (
            ce_weight.shape[0] == V
        ), f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(
            ce_weight
        ), f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask))
            .sum()
            .item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]

        logits_chunk = _input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]

        n_rows = logits_chunk.shape[0]

        loss_1d_slice = loss_1d[start_idx:end_idx]
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=ce_weight,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
            weight_sum=ce_weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        loss_1d[start_idx:end_idx] = loss_1d_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice
        grad_logits_chunk = logits_chunk

        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=logits_chunk.t().to(_input_chunk.dtype),
                mat2=_input_chunk,
                out=grad_weight,
                alpha=1.0,
                beta=1.0,
            )

        if bias is not None:
            torch.add(
                input=grad_bias,
                other=logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
    return loss, z_loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(
    grad_output, grad_input, grad_weight, grad_bias
):

    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):

        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
    ):

        loss, z_loss, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                ce_weight=ce_weight,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                return_z_loss=return_z_loss,
            )
        )

        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
