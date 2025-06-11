import torch
import triton
import triton.language as tl


@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q,
    K,
    V,
    seqlens,
    sm_scale,
    block_count,
    block_offset,
    column_count,
    column_index,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    NUM_ROWS,
    NNZ_S,
    NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    m_mask = offs_m[:, None] < seqlen

    max_num_blks = 8
    for block_index in range(max_num_blks):

        cond = block_index < num_blks

        start_n = tl.load(blks_ptr + block_index, mask=cond)
        cols = start_n + offs_n
        n_mask = (cols < seqlen) & cond[:, None]
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    max_num_cols = 16
    for start_n in range(0, max_num_cols, BLOCK_N):

        cond = start_n < num_cols
        n_mask = (start_n + offs_n < num_cols) & cond
        cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_mixed_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q,
        k,
        v,
        seqlens,
        sm_scale,
        block_count,
        block_offset,
        column_count,
        column_index,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        block_count.shape[-1],
        block_offset.shape[-1],
        column_index.shape[-1],
        BLOCK_M=block_size_M,
        BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4,
        num_stages=2,
    )

    return o


import torch


def test_triton_mixed_sparse_attention():

    batch_size = 2
    num_heads = 4
    seq_len = 128
    d_model = 64
    block_size_M = 64
    block_size_N = 64
    sm_scale = 0.1

    q = torch.randn(
        (batch_size, num_heads, seq_len, d_model), dtype=torch.float16, device="cuda"
    )
    k = torch.randn(
        (batch_size, num_heads, seq_len, d_model), dtype=torch.float16, device="cuda"
    )
    v = torch.randn(
        (batch_size, num_heads, seq_len, d_model), dtype=torch.float16, device="cuda"
    )
    seqlens = torch.randint(
        low=1, high=seq_len, size=(batch_size,), dtype=torch.int32, device="cuda"
    )

    block_count = torch.randint(
        low=1,
        high=seq_len // block_size_M,
        size=(batch_size, num_heads, seq_len // block_size_M),
        dtype=torch.int32,
        device="cuda",
    )
    block_offset = torch.randint(
        low=0,
        high=seq_len,
        size=(batch_size, num_heads, seq_len // block_size_M, 4),
        dtype=torch.int32,
        device="cuda",
    )
    column_count = torch.randint(
        low=1,
        high=seq_len // block_size_N,
        size=(batch_size, num_heads, seq_len // block_size_M),
        dtype=torch.int32,
        device="cuda",
    )
    column_index = torch.randint(
        low=0,
        high=seq_len,
        size=(batch_size, num_heads, seq_len // block_size_M, 8),
        dtype=torch.int32,
        device="cuda",
    )

    output1 = _triton_mixed_sparse_attention(
        q,
        k,
        v,
        seqlens,
        block_count,
        block_offset,
        column_count,
        column_index,
        sm_scale,
        block_size_M,
        block_size_N,
    )

    block_size_M_alt = 32
    block_size_N_alt = 32
    output2 = _triton_mixed_sparse_attention(
        q,
        k,
        v,
        seqlens,
        block_count,
        block_offset,
        column_count,
        column_index,
        sm_scale,
        block_size_M_alt,
        block_size_N_alt,
    )

    sm_scale_alt = 0.2
    output3 = _triton_mixed_sparse_attention(
        q,
        k,
        v,
        seqlens,
        block_count,
        block_offset,
        column_count,
        column_index,
        sm_scale_alt,
        block_size_M,
        block_size_N,
    )

    seqlens_alt = torch.randint(
        low=1, high=seq_len, size=(batch_size,), dtype=torch.int32, device="cuda"
    )
    output4 = _triton_mixed_sparse_attention(
        q,
        k,
        v,
        seqlens_alt,
        block_count,
        block_offset,
        column_count,
        column_index,
        sm_scale,
        block_size_M,
        block_size_N,
    )

    return {
        "test_case_1": output1,
        "test_case_2": output2,
        "test_case_3": output3,
        "test_case_4": output4,
    }


result_gold = test_triton_mixed_sparse_attention()
