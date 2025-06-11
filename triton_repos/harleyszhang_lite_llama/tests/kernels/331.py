import triton, torch
import triton.language as tl


def naive_softmax(x: torch.Tensor) -> torch.Tensor:

    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]

    return ret


def online_softmax(x: torch.Tensor) -> torch.tensor:

    row_cont, col_count = x.shape
    assert x.ndim == 2, f"only accepts 2D tensor now"
    output = torch.zeros_like(x)

    for r in range(row_cont):
        row_max = x[r][0]
        normalizer = 0
        for c in range(1, col_count):
            pre_max = row_max
            cur = x[r][c]
            row_max = max(pre_max, cur)

            normalizer = normalizer * torch.exp(pre_max - row_max) + torch.exp(
                cur - row_max
            )
        output[r, :] = torch.exp(x[r, :] - row_max) / normalizer

    return output


@triton.jit
def _softmax_kernel_fwd(
    input_ptr,
    stride_input_row,
    output_ptr,
    stride_output_row,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):

    row_id = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_id * stride_input_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_pointers = row_start_ptr + col_offsets

    row_data_mask = col_offsets < num_cols

    x = tl.load(input_pointers, mask=row_data_mask, other=0.0)

    safe_row = x - tl.max(x, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    output_row_ptr = output_ptr + row_id * stride_input_row
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, softmax_out, mask=row_data_mask)


@torch.no_grad()
def softmax_native_fwd(x: torch.Tensor) -> torch.Tensor:

    rows, cols = x.shape
    assert x.ndim == 2, f"only accepts 2D tensor now"
    BLOCK_SIZE = triton.next_power_of_2(cols)
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    grid = (rows, 1)

    softmax_out = torch.empty_like(x)

    _softmax_kernel_fwd[grid](
        x,
        x.stride(0),
        softmax_out,
        softmax_out.stride(0),
        cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return softmax_out
