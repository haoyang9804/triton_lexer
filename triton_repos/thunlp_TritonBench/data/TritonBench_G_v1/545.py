import torch
import triton
import triton.language as tl


@triton.jit
def index_select_cat_fwd_kernel(
    output_ptr,
    source_ptr,
    index_ptr,
    num_indices,
    num_cols,
    stride0,
    stride1,
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    rows = tl.load(index_ptr + indices, mask=(indices < num_indices))
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)

    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)


def index_select_cat_fwd(
    output: torch.Tensor,
    source: torch.Tensor,
    index: torch.Tensor,
):
    if not (source.is_cuda and index.is_cuda):
        raise ValueError("The index tensor and the source tensor must be of type CUDA!")

    if not source.ndim == 2:
        raise ValueError(f"Expected 2-dimensional tensor, got {source.ndim}.")
    if not index.ndim == 1:
        raise ValueError(f"Expected 1-dimensional tensor, got {index.ndim}.")

    num_rows, num_cols = source.shape
    num_indices = index.shape[0]

    if num_indices > num_rows:
        print(
            f"Warning: The number of indices exceeds the number of rows in the source tensor. Truncating indices."
        )
        num_indices = num_rows
        index = index[:num_rows]

    stride0, stride1 = source.stride(0), source.stride(1)

    def grid(meta):
        return (
            triton.cdiv(num_indices, meta["BLOCK_SIZE_INDEX"]),
            triton.cdiv(num_cols, meta["BLOCK_SIZE_COL"]),
        )

    index_select_cat_fwd_kernel[grid](
        output,
        source,
        index,
        num_indices,
        num_cols,
        stride0,
        stride1,
        BLOCK_SIZE_INDEX=1,
        BLOCK_SIZE_COL=512,
    )

    return output


def test_index_select_cat_fwd():
    results = {}

    source = torch.randn(10, 512, device="cuda")
    index = torch.tensor([0, 2, 4, 6, 8], device="cuda")
    output = torch.empty(len(index), source.size(1), device="cuda")
    index_select_cat_fwd(output, source, index)
    results["test_case_1"] = output.clone()

    index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device="cuda")
    output = torch.empty(len(index), source.size(1), device="cuda")
    index_select_cat_fwd(output, source, index)
    results["test_case_2"] = output.clone()

    index = torch.tensor([0], device="cuda")
    output = torch.empty(len(index), source.size(1), device="cuda")
    index_select_cat_fwd(output, source, index)
    results["test_case_3"] = output.clone()

    index = torch.tensor([9, 7, 5, 3, 1], device="cuda")
    output = torch.empty(len(index), source.size(1), device="cuda")
    index_select_cat_fwd(output, source, index)
    results["test_case_4"] = output.clone()

    return results


result_gold = test_index_select_cat_fwd()
