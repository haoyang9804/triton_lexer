import torch
import triton

import triton.language as tl


@triton.jit
def vector_addition_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    x_stride: tl.constexpr,
    y_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x_offsets = offsets * x_stride
    y_offsets = offsets * y_stride

    mask = offsets < n_elements

    x = tl.load(x_ptr + x_offsets, mask=mask)
    y = tl.load(y_ptr + y_offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + x_offsets, output, mask=mask)


def vector_addition(x, y):

    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU!"

    assert x.numel() == y.numel(), "Input tensors must be the same size!"

    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

    vector_addition_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=x.numel(),
        x_stride=1,
        y_stride=1,
        output_stride=1,
        BLOCK_SIZE=128,
    )

    return output


if __name__ == "__main__":

    x = torch.randn(1024, device="cuda")
    y = torch.randn(1024, device="cuda")

    output = vector_addition(x, y)

    output_ref = x + y
    assert torch.allclose(output, output_ref)
    print("Success with power of 2 size (1024!")

    print(f"{output=}")

    x = torch.randn(257, device="cuda")
    y = torch.randn(257, device="cuda")

    output_np2 = vector_addition(x, y)

    output_ref_np2 = x + y
    assert torch.allclose(output_np2, output_ref_np2)
    print("Success with non power of 2 size (num_elems = 257!)")

    print(f"{output_np2[0:5]=}")
