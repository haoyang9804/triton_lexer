import triton
import triton.language as tl

from .utils import element_wise_kernel_configs


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=["size"],
)
@triton.jit
def p_loss_forward_kernel(
    input_pointer,
    target_pointer,
    output_pointer,
    param,
    size,
    p_loss: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input = tl.load(input_pointer + offset, mask=mask).to(tl.float32)
    target = tl.load(target_pointer + offset, mask=mask).to(tl.float32)
    diff = input - target

    if p_loss == 0:
        error = tl.where(
            diff < param, 0.5 * diff * diff / param, tl.abs(diff) - 0.5 * param
        )

    elif p_loss == 1:
        error = tl.abs(diff)

    elif p_loss == 2:
        error = diff * diff

    elif p_loss == 3:
        error = tl.where(
            diff < param, 0.5 * diff * diff, param * (tl.abs(diff) - 0.5 * param)
        )

    if reduction == "none":
        tl.store(output_pointer + offset, error, mask=mask)

    elif reduction == "mean":
        tl.store(output_pointer + pid, tl.sum(error) / size)

    elif reduction == "sum":
        tl.store(output_pointer + pid, tl.sum(error))


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=["size"],
)
@triton.jit
def p_loss_backward_kernel(
    output_grad_pointer,
    input_pointer,
    target_pointer,
    input_grad_pointer,
    target_grad_pointer,
    param,
    size,
    p_loss: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad_mask = None
    if reduction == "none":
        output_grad_pointer += offset
        output_grad_mask = mask

    input = tl.load(input_pointer + offset, mask=mask).to(tl.float32)
    target = tl.load(target_pointer + offset, mask=mask).to(tl.float32)
    diff = input - target
    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask).to(tl.float32)

    if p_loss == 0:
        input_grad = tl.where(diff < param, diff / param, tl.where(0 <= diff, 1, -1))

    elif p_loss == 1:
        input_grad = tl.where(0 <= diff, 1, -1)

    elif p_loss == 2:
        input_grad = 2 * diff

    elif p_loss == 3:
        input_grad = tl.where(diff < param, diff, param * tl.where(0 <= diff, 1, -1))

    if reduction == "mean":
        input_grad /= size

    input_grad *= output_grad
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)
    tl.store(target_grad_pointer + offset, -input_grad, mask=mask)
