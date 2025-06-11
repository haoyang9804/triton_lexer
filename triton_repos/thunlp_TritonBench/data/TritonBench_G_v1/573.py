import torch
import triton
import triton.language as tl


@triton.jit
def _bgmv_expand_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_indices,
    xm_stride,
    xk_stride,
    l0_stride,
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):

    pid_sn = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    if EVEN_K:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
        )
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )

    split_n_length = tl.cdiv(N, SPLIT_N)
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)

    b_ptr = lora_ptr + l0_stride * lora_index + pid_sn * split_n_length * lora_k_stride
    c_ptr = out_ptr + cur_batch * cm_stride + pid_sn * split_n_length
    for n in range(0, split_n_length, BLOCK_N):
        current_n = n + offset_n
        current_n_c = tl.max_contiguous(current_n, BLOCK_N)
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :] < K)
        c_mask = current_n < split_n_length
        tiled_b = tl.load(
            b_ptr
            + current_n_c[:, None] * lora_k_stride
            + offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)

        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)


@torch.inference_mode()
def _bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
) -> None:

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)

    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    if lora_b_weights.ndim == 4:
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3
    assert lora_b_weights.is_contiguous()

    N, K = lora_b_weights.shape[-2:]
    BLOCK_K = triton.next_power_of_2(K)
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]:
        CAST_TYPE = True
    batches = lora_indices_tensor.size(0)

    grid = lambda META: (
        META["SPLIT_N"],
        batches,
    )
    _bgmv_expand_kernel[grid](
        inputs,
        lora_b_weights,
        output_tensor,
        N,
        K,
        lora_indices_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_b_weights.stride(0),
        lora_b_weights.stride(1),
        lora_b_weights.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_K=BLOCK_K,
        BLOCK_N=256,
        SPLIT_N=lora_b_weights.shape[-2:][0],
        EVEN_K=EVEN_K,
        ADD_INPUTS=ADD_INPUTS,
        CAST_TYPE=CAST_TYPE,
    )
    return


import torch


def test_bgmv_expand():

    batch_size = 4
    hidden_size = 128
    rank = 64
    lora_num = 3

    inputs = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="cuda")
    lora_b_weights = torch.randn(
        lora_num, rank, hidden_size, dtype=torch.float16, device="cuda"
    )
    lora_indices_tensor = torch.tensor([0, 1, -1, 2], dtype=torch.int32, device="cuda")

    results = {}

    output_tensor_1 = torch.zeros(batch_size, rank, dtype=torch.float16, device="cuda")
    _bgmv_expand(
        inputs=inputs,
        lora_b_weights=lora_b_weights,
        output_tensor=output_tensor_1,
        lora_indices_tensor=lora_indices_tensor,
        add_inputs=True,
    )
    results["test_case_1"] = output_tensor_1

    output_tensor_2 = torch.zeros(batch_size, rank, dtype=torch.float16, device="cuda")
    _bgmv_expand(
        inputs=inputs,
        lora_b_weights=lora_b_weights,
        output_tensor=output_tensor_2,
        lora_indices_tensor=lora_indices_tensor,
        add_inputs=False,
    )
    results["test_case_2"] = output_tensor_2

    return results


result_gold = test_bgmv_expand()
