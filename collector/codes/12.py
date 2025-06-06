import triton
import triton.language as tl
from triton import next_power_of_2

from .softmax_kernels import BLOCK_SIZE_BATCH_heuristic
from .utils import warps_kernel_configs


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["batch_dim", "feat_dim"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH_heuristic,
        "BLOCK_SIZE_FEAT": lambda args: next_power_of_2(args["feat_dim"]),
    }
)
@triton.jit
def layer_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    batch_dim,
    feat_dim,
    input_batch_stride,
    input_feat_stride,
    output_batch_stride,
    output_feat_stride,
    eps,
    scale_by_weight: tl.constexpr,
    add_bias: tl.constexpr,
    save_stats: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):

    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    input_pointer += (
        input_batch_stride * batch_offset[:, None]
        + input_feat_stride * feat_offset[None, :]
    )
    output_pointer += (
        output_batch_stride * batch_offset[:, None]
        + output_feat_stride * feat_offset[None, :]
    )

    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :]).to(
        tl.float32
    )
    mean = tl.sum(input, axis=1) / feat_dim
    diff = tl.where(feat_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / feat_dim + eps)

    if save_stats:
        tl.store(mean_pointer + batch_offset, mean, mask=batch_mask)
        tl.store(inv_std_pointer + batch_offset, inv_std, mask=batch_mask)

    output = diff * inv_std[:, None]
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        output *= weight
        if add_bias:
            bias = tl.load(bias_pointer + feat_offset, mask=feat_mask)
            output += bias

    tl.store(output_pointer, output, mask=batch_mask[:, None] & feat_mask[None, :])


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["batch_dim", "feat_dim"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH_heuristic,
        "BLOCK_SIZE_FEAT": lambda args: next_power_of_2(args["feat_dim"]),
    }
)
@triton.jit
def layer_norm_backward_kernel(
    output_grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    feat_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    input_batch_stride,
    input_feat_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    weight_grad_batch_stride,
    weight_grad_feat_stride,
    bias_grad_batch_stride,
    bias_grad_feat_stride,
    scale_by_weight: tl.constexpr,
    add_bias: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):

    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    output_grad_pointer += (
        output_grad_batch_stride * batch_offset[:, None]
        + output_grad_feat_stride * feat_offset[None, :]
    )
    input_pointer += (
        input_batch_stride * batch_offset[:, None]
        + input_feat_stride * feat_offset[None, :]
    )
    input_grad_pointer += (
        input_grad_batch_stride * batch_offset[:, None]
        + input_grad_feat_stride * feat_offset[None, :]
    )

    output_grad = tl.load(
        output_grad_pointer, mask=batch_mask[:, None] & feat_mask[None, :]
    ).to(tl.float32)
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :]).to(
        tl.float32
    )
    mean = tl.load(mean_pointer + batch_offset, mask=batch_mask)
    inv_std = tl.load(inv_std_pointer + batch_offset, mask=batch_mask)
    pre_lin = (input - mean[:, None]) * inv_std[:, None]

    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        weight_output_grad_prod = weight * output_grad

    else:
        weight_output_grad_prod = output_grad

    term1 = tl.sum(pre_lin * weight_output_grad_prod, axis=1) / feat_dim
    term1 = pre_lin * term1[:, None]
    term2 = tl.sum(weight_output_grad_prod, axis=1) / feat_dim
    input_grad = inv_std[:, None] * (weight_output_grad_prod - (term1 + term2[:, None]))

    tl.store(
        input_grad_pointer, input_grad, mask=batch_mask[:, None] & feat_mask[None, :]
    )

    if scale_by_weight:
        weight_grad_pointer += (
            weight_grad_batch_stride * batch_pid + weight_grad_feat_stride * feat_offset
        )
        tl.store(
            weight_grad_pointer, tl.sum(output_grad * pre_lin, axis=0), mask=feat_mask
        )

        if add_bias:
            bias_grad_pointer += (
                bias_grad_batch_stride * batch_pid + bias_grad_feat_stride * feat_offset
            )
            tl.store(bias_grad_pointer, tl.sum(output_grad, axis=0), mask=feat_mask)
