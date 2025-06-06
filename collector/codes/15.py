import triton
import triton.language as tl

from .utils import element_wise_kernel_configs


@triton.jit
def apply_dropout(input, drop_p, seed, offset):

    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))


@triton.jit
def apply_dropout_grad(output_grad, drop_p, seed, offset):

    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, output_grad / (1 - drop_p))


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=["size"],
)
@triton.jit
def dropout_forward_kernel(
    input_pointer,
    output_pointer,
    size,
    drop_p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input = tl.load(input_pointer + offset, mask=mask)
    output = apply_dropout(input, drop_p, seed, offset)
    tl.store(output_pointer + offset, output, mask=mask)


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=["size"],
)
@triton.jit
def dropout_backward_kernel(
    output_grad_pointer,
    input_grad_pointer,
    size,
    drop_p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)
