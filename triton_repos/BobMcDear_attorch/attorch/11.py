import triton
import triton.language as tl
from triton import next_power_of_2

from .utils import warps_kernel_configs


def BLOCK_SIZE_BATCH_heuristic(args) -> int:

    return (
        min(max(1, next_power_of_2(args["batch_dim"] // 2**10)), 128)
        if args["spatial_dim"] < 64
        else 1
    )


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH_heuristic,
        "BLOCK_SIZE_SPATIAL": lambda args: next_power_of_2(args["spatial_dim"]),
    }
)
@triton.jit
def nll_loss_forward_kernel(
    input_pointer,
    target_pointer,
    weight_pointer,
    sum_weights_pointer,
    output_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    target_batch_stride,
    target_spatial_stride,
    output_batch_stride,
    output_spatial_stride,
    reduction: tl.constexpr,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):

    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)

    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim

    target_pointer += (
        target_batch_stride * batch_offset[:, None]
        + target_spatial_stride * spatial_offset[None, :]
    )
    target = tl.load(target_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])

    input_pointer += (
        input_feat_stride * target
        + input_batch_stride * batch_offset[:, None]
        + input_spatial_stride * spatial_offset[None, :]
    )
    input = tl.load(input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]).to(
        tl.float32
    )

    output = -input
    if weighted:
        weight = tl.load(
            weight_pointer + target, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        output *= weight

    if reduction == "none":
        output_pointer += (
            output_batch_stride * batch_offset[:, None]
            + output_spatial_stride * spatial_offset[None, :]
        )
        tl.store(
            output_pointer, output, mask=batch_mask[:, None] & spatial_mask[None, :]
        )

    elif reduction == "mean":
        if weighted:
            tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))
            tl.store(output_pointer + batch_pid, tl.sum(output))

        else:
            tl.store(
                output_pointer + batch_pid, tl.sum(output) / (batch_dim * spatial_dim)
            )

    elif reduction == "sum":
        tl.store(output_pointer + batch_pid, tl.sum(output))


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH_heuristic,
        "BLOCK_SIZE_SPATIAL": lambda args: next_power_of_2(args["spatial_dim"]),
    }
)
@triton.jit
def nll_loss_backward_kernel(
    output_grad_pointer,
    target_pointer,
    weight_pointer,
    sum_weights_pointer,
    input_grad_pointer,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    target_batch_stride,
    target_spatial_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    reduction: tl.constexpr,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):

    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)

    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim

    output_grad_mask = None
    if reduction == "none":
        output_grad_pointer += (
            output_grad_batch_stride * batch_offset[:, None]
            + output_grad_feat_stride * spatial_offset[None, :]
        )
        output_grad_mask = batch_mask[:, None] & spatial_mask[None, :]

    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask).to(tl.float32)
    input_grad = -output_grad

    target_pointer += (
        target_batch_stride * batch_offset[:, None]
        + target_spatial_stride * spatial_offset[None, :]
    )
    target = tl.load(target_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])

    if weighted:
        weight = tl.load(
            weight_pointer + target, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        input_grad *= weight

        if reduction == "mean":
            input_grad /= tl.load(sum_weights_pointer)

    elif reduction == "mean":
        input_grad /= batch_dim * spatial_dim

    input_grad_pointer += (
        input_grad_feat_stride * target
        + input_grad_batch_stride * batch_offset[:, None]
        + input_grad_spatial_stride * spatial_offset[None, :]
    )
    tl.store(
        input_grad_pointer, input_grad, mask=batch_mask[:, None] & spatial_mask[None, :]
    )
