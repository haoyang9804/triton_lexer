import torch, math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd
from typing import List, Optional, Union
import torch.nn.functional as F


@triton.jit
def flash_attention_v1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_batch_stride,
    q_heads_stride,
    q_seq_stride,
    q_dim_stride,
    k_batch_stride,
    k_heads_stride,
    k_seq_stride,
    k_dim_stride,
    v_batch_stride,
    v_heads_stride,
    v_seq_stride,
    v_dim_stride,
    out_batch_stride,
    out_heads_stride,
    out_seq_stride,
    out_dim_stride,
    num_kv_groups,
    n_heads,
    m_size,
    n_size,
    BLOCK_DHEAD_SIZE: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    sm_scale,
    causal_mask,
):

    block_m_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    cur_batch_idx = head_idx // n_heads
    cur_head_idx = head_idx % n_heads

    cur_kv_head_idx = cur_head_idx // num_kv_groups

    m_range_offs = tl.arange(0, BLOCK_M_SIZE)
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)
    dhead_range_offs = tl.arange(0, BLOCK_DHEAD_SIZE)

    m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs

    q_offs = (
        cur_batch_idx * q_batch_stride
        + cur_head_idx * q_heads_stride
        + (m_offs[:, None] * q_seq_stride + dhead_range_offs[None, :] * q_dim_stride)
    )

    k_offs = (
        cur_batch_idx * k_batch_stride
        + cur_kv_head_idx * k_heads_stride
        + (
            n_range_offs[:, None] * k_seq_stride
            + dhead_range_offs[None, :] * k_dim_stride
        )
    )

    v_offs = (
        cur_batch_idx * v_batch_stride
        + cur_kv_head_idx * v_heads_stride
        + (
            n_range_offs[:, None] * v_seq_stride
            + dhead_range_offs[None, :] * v_dim_stride
        )
    )

    o_offs = (
        cur_batch_idx * out_batch_stride
        + cur_head_idx * out_heads_stride
        + (
            m_offs[:, None] * out_seq_stride
            + dhead_range_offs[None, :] * out_dim_stride
        )
    )

    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs
    v_ptrs = v_ptr + v_offs
    out_ptrs = o_ptr + o_offs

    l_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)

    q_mask = m_offs[:, None] < m_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
        block_n_offs = block_n_start_idx + n_range_offs
        k_mask = block_n_offs[:, None] < n_size
        k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask, other=0.0)

        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if causal_mask:
            offs_k = block_n_offs
            offs_m = m_offs

            mask = offs_m[:, None] >= offs_k[None, :]

            qk = tl.where(mask, qk * sm_scale, -1.0e8)
        else:
            qk = qk * sm_scale

        l_j = tl.max(qk, 1)
        numerators = tl.exp(qk - l_j[:, None])
        d_j = tl.sum(numerators, 1)

        l_new = tl.maximum(l_i, l_j)
        alpha = tl.exp(l_i - l_new)
        beta = tl.exp(l_j - l_new)
        d_new = alpha * d_i + beta * d_j

        p_scale = beta / d_new
        p = numerators * p_scale[:, None]

        sigma = d_i / d_new * alpha
        acc = acc * sigma[:, None]

        v = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=k_mask, other=0.0)
        p = p.to(q_ptr.dtype.element_ty)

        acc += tl.dot(p, v)

        l_i = l_new
        d_i = d_new

    out_mask = m_offs[:, None] < m_size
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):

    num_kv_groups = q.shape[1] // k.shape[1]
    output = torch.empty_like(q)
    assert q.device.type == "cuda", "Input tensor q must be on CUDA device"
    assert k.device.type == "cuda", "Input tensor keys must be on CUDA device"

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"

    bs, n_heads, m_size, HEAD_DIM = q.size()
    causal_mask = False
    if m_size > 1:
        causal_mask: bool = True

    n_size = k.shape[2]
    sm_scale = 1 / math.sqrt(HEAD_DIM)

    grid = lambda meta: (
        triton.cdiv(m_size, meta["BLOCK_M_SIZE"]),
        bs * n_heads,
        1,
    )

    flash_attention_v1_kernel[grid](
        q,
        k,
        v,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        num_kv_groups,
        n_heads,
        m_size,
        n_size,
        HEAD_DIM,
        32,
        32,
        sm_scale,
        causal_mask,
    )
    return output


def standard_attention(Q, K, V, sm_scale, mask=None):

    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(attn_scores, dim=-1)

    out = torch.matmul(attn_weights, V)

    return out


def test_prefill_stage():

    batch_size = 2
    num_heads = 4
    seq_length = 32
    head_dim = 32
    BLOCK_M = 32
    BLOCK_N = 32

    torch.manual_seed(0)
    q = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )
    k = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )
    v = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )

    sm_scale = 1.0 / math.sqrt(head_dim)

    out = flash_attention_v1(q, k, v)

    mask = (
        torch.tril(torch.ones((seq_length, seq_length)))
        .unsqueeze(0)
        .unsqueeze(0)
        .type_as(q)
    )
    standard_o = standard_attention(q, k, v, sm_scale, mask)

    if torch.allclose(out, standard_o, atol=1e-2):
        print(
            "Prefill Stage Test Passed: Triton output matches PyTorch standard implementation."
        )
    else:
        max_diff = (out - standard_o).abs().max()
        print(f"Prefill Stage Test Failed: Maximum difference {max_diff}")


def test_decode_stage():

    batch_size = 1
    num_heads = 4
    initial_seq_length = 16
    generated_seq_length = 16
    head_dim = 64
    BLOCK_M = 16
    BLOCK_N = 16

    torch.manual_seed(0)
    q_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    k_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    v_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    o_initial = torch.zeros_like(q_initial, device="cuda", dtype=torch.float32)
    new_token_q = torch.randn(
        batch_size, num_heads, 1, head_dim, device="cuda", dtype=torch.float32
    )

    triton_k_extended = k_initial
    triton_v_extended = v_initial
    torch_k_extended = k_initial
    torch_v_extended = v_initial
    torch_new_token_q = new_token_q
    triton_new_token_q = new_token_q

    for step in range(1, generated_seq_length + 1):

        triton_k_extended = torch.cat([triton_k_extended, triton_new_token_q], dim=2)
        triton_v_extended = torch.cat([triton_v_extended, triton_new_token_q], dim=2)

        torch_k_extended = torch.cat([torch_k_extended, torch_new_token_q], dim=2)
        torch_v_extended = torch.cat([torch_v_extended, torch_new_token_q], dim=2)

        sm_scale_extended = 1.0 / math.sqrt(head_dim)

        triton_new_token_q = flash_attention_v1(
            new_token_q, triton_k_extended, triton_v_extended
        )

        torch_new_token_q = standard_attention(
            new_token_q, torch_k_extended, torch_v_extended, sm_scale_extended
        )

        if torch.allclose(triton_new_token_q, torch_new_token_q, atol=1e-1):
            print(
                f"Decode Stage Step {step} Test Passed: Triton output matches PyTorch standard implementation."
            )
        else:
            max_diff = (triton_new_token_q - torch_new_token_q).abs().max()
            print(
                f"Decode Stage Step {step} Test Failed: Maximum difference {max_diff}"
            )

            break


if __name__ == "__main__":
    print("Running Prefill Stage Test...")
    test_prefill_stage()
    print("\nRunning Decode Stage Test...")
    test_decode_stage()
