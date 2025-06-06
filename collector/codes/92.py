import torch
import triton
import triton.language as tl
import triton_viz
import argparse
from triton_viz.interpreter import record_builder
import numpy as np
from triton_viz.data import Load


@triton_viz.trace
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = tl.zeros(x.shape, dtype=x.dtype)
    output = output + x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, access_size: int, BLOCK_SIZE: int = 1024):

    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(access_size, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, access_size, BLOCK_SIZE=BLOCK_SIZE)

    return output, grid


def perform_vec_add(device, size, access_size=None):
    torch.manual_seed(0)
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    access_size = size if access_size is None else access_size
    output, _ = add(x, y, access_size=access_size)
    return x, y, output


def test_add():
    device = "cpu"
    size = 5000
    BLOCK_SIZE = 1024
    input_vector1, input_vector2, result = perform_vec_add(device, size)
    t_size = input_vector1.element_size()
    expected_offsets = [i * t_size for i in np.arange(0, BLOCK_SIZE)]
    expected_offsets_len = len(expected_offsets)
    expected = input_vector1 + input_vector2
    expected_masks = np.ones(expected_offsets_len, dtype=bool)
    expected_invalid_masks = np.logical_not(expected_masks)
    for op in record_builder.launches[0].records:
        if isinstance(op, Load):
            result_offsets = op.offsets.tolist()
            result_offsets_len = len(result_offsets)
            result_masks = op.access_masks
            result_invalid_masks = op.invalid_access_masks
            break
    assert torch.allclose(result, expected)
    assert result.shape == expected.shape
    assert result_offsets == expected_offsets
    assert result_offsets_len == expected_offsets_len
    assert (result_masks == expected_masks).all()
    assert (result_invalid_masks == expected_invalid_masks).all()


def test_out_of_bounds_add():
    device = "cpu"
    size = 960
    BLOCK_SIZE = 1024
    input_vector1, input_vector2, result = perform_vec_add(
        device, size=size, access_size=BLOCK_SIZE
    )
    t_size = input_vector1.element_size()
    expected_offsets = [(i * t_size) if i < size else 0 for i in range(BLOCK_SIZE)]
    expected_offsets_len = len(expected_offsets)
    expected = input_vector1 + input_vector2
    expected_original_masks = [True for i in range(BLOCK_SIZE)]
    expected_valid_masks = [i < size for i in range(BLOCK_SIZE)]
    expected_invalid_masks = [i >= size for i in range(BLOCK_SIZE)]
    for op in record_builder.launches[0].records:
        if isinstance(op, Load):
            result_offsets = op.offsets.tolist()
            result_offsets_len = len(result_offsets)
            result_original_masks = op.original_masks
            result_valid_masks = op.access_masks
            result_invalid_masks = op.invalid_access_masks
            break
    assert torch.allclose(result, expected)
    assert result.shape == expected.shape
    assert result_offsets == expected_offsets
    assert result_offsets_len == expected_offsets_len
    assert (result_original_masks == expected_original_masks).all()
    assert (result_valid_masks == expected_valid_masks).all()
    assert (result_invalid_masks == expected_invalid_masks).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    device = args.device

    size = 5000
    input_vector1, input_vector2, output_triton = perform_vec_add(device, size)
    triton_viz.launch()
