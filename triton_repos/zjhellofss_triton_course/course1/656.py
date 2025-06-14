import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):

    row_len = 2
    row_start = tl.program_id(0) * row_len
    if row_start >= n_rows:
        return
    for row_idx in tl.range(row_start, row_start + row_len, 1):

        row_start_ptr = input_ptr + row_idx * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        row_minus_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


input_tensor = torch.randn(1000, 512, device="cuda")
output_tensor = torch.empty_like(input_tensor)


n_rows, n_cols = input_tensor.shape
BLOCK_SIZE = triton.next_power_of_2(n_cols)
num_stages = 3


grid = lambda meta: (triton.cdiv(n_rows, 2),)
softmax_kernel[grid](
    output_tensor,
    input_tensor,
    input_tensor.stride(0),
    output_tensor.stride(0),
    n_rows,
    n_cols,
    BLOCK_SIZE=BLOCK_SIZE,
)


expected_output = torch.softmax(input_tensor, dim=1)
print(
    "Triton Softmax 和 PyTorch Softmax 是否接近:",
    torch.allclose(output_tensor, expected_output, atol=1e-6),
)
