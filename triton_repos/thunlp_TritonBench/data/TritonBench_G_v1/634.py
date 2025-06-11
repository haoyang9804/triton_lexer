import triton
import triton.language as tl
import torch


@triton.jit
def square_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))

    square_output = row * row

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)


def square(x):
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)

    square_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


import torch


def test_square():
    x_triton_1 = torch.randn(128, 64, device="cuda")
    x_triton_2 = torch.randn(128, 128, device="cuda")
    x_triton_3 = torch.randn(128, 256, device="cuda")
    x_triton_4 = torch.randn(128, 512, device="cuda")

    y_triton_1 = square(x_triton_1)
    y_triton_2 = square(x_triton_2)
    y_triton_3 = square(x_triton_3)
    y_triton_4 = square(x_triton_4)

    return {
        "test_case_1": y_triton_1,
        "test_case_2": y_triton_2,
        "test_case_3": y_triton_3,
        "test_case_4": y_triton_4,
    }


result_gold = test_square()
