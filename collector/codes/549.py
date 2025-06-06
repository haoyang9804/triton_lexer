import torch

import triton
import triton.language as tl
import torch.nn.functional as F


def naive_softmax(x: torch.Tensor) -> torch.Tensor:

    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    sm_out = numerator / denominator[:, None]
    return sm_out


@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
):

    row_index = tl.program_id(0)

    row_start_ptr = input_ptr + (row_index * stride_input_row)
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols

    row = tl.load(input_pointers, mask=row_mask, other=float("-inf"))

    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)


def softmax(x: torch.Tensor) -> torch.Tensor:

    rows, cols = x.shape
    assert x.dim() == 2, f"only accepts 2D tensors for now"
    block_size = triton.next_power_of_2(cols)
    num_warps = 4
    if block_size > 2047:
        num_warps = 8
    if block_size > 4095:
        num_warps = 16

    grid = (rows,)

    sm_out = torch.empty_like(x)

    _softmax_fwd_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps,
    )

    return sm_out


sample = torch.tensor(
    [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device="cuda"
)
ref_out = F.softmax(sample, dim=1)
print(f"{ref_out=}")


eager_out = naive_softmax(sample)
print(f"{eager_out=}")

triton_out = softmax(sample)
print(f"{triton_out=}")
