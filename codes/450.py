import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc,
    m_i,
    d_i,
    q,
    k_ptrs,
    v_ptrs,
    k_seq_stride,
    v_seq_stride,
    offs_m,
    qk_scale,
    n_size,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    fp8_v: tl.constexpr,
):
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)

    for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
        block_n_start_idx = tl.multiple_of(block_n_start_idx, BLOCK_N_SIZE)
        block_n_offs = block_n_start_idx + n_range_offs

        k_mask = block_n_offs[:, None] < n_size
        k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k))

        offs_k = block_n_offs

        mask = offs_m[:, None] >= offs_k[None, :]
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        d_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        d_i = d_i * alpha + d_ij

        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=k_mask, other=0.0)
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc)

        m_i = m_ij

    return acc, d_i


@triton.jit
def flash_attention_v2_kernel(
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
    HEAD_DIM: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    qk_scale,
):

    block_m_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    cur_batch_idx = head_idx // n_heads
    cur_head_idx = head_idx % n_heads

    cur_kv_head_idx = cur_head_idx // num_kv_groups

    m_range_offs = tl.arange(0, BLOCK_M_SIZE)
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)
    dhead_range_offs = tl.arange(0, HEAD_DIM)

    offs_m = block_m_idx * BLOCK_M_SIZE + m_range_offs

    offs_q = (
        cur_batch_idx * q_batch_stride
        + cur_head_idx * q_heads_stride
        + (offs_m[:, None] * q_seq_stride + dhead_range_offs[None, :] * q_dim_stride)
    )

    offs_k = (
        cur_batch_idx * k_batch_stride
        + cur_kv_head_idx * k_heads_stride
        + (
            n_range_offs[:, None] * k_seq_stride
            + dhead_range_offs[None, :] * k_dim_stride
        )
    )

    offs_v = (
        cur_batch_idx * v_batch_stride
        + cur_kv_head_idx * v_heads_stride
        + (
            n_range_offs[:, None] * v_seq_stride
            + dhead_range_offs[None, :] * v_dim_stride
        )
    )

    offs_o = (
        cur_batch_idx * out_batch_stride
        + cur_head_idx * out_heads_stride
        + (
            offs_m[:, None] * out_seq_stride
            + dhead_range_offs[None, :] * out_dim_stride
        )
    )

    q_ptrs = q_ptr + offs_q
    k_ptrs = k_ptr + offs_k
    v_ptrs = v_ptr + offs_v
    out_ptrs = o_ptr + offs_o

    q_mask = offs_m[:, None] < m_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.zeros(
        [
            BLOCK_M_SIZE,
        ],
        dtype=tl.float32,
    ) - float("inf")
    d_i = tl.zeros(
        [
            BLOCK_M_SIZE,
        ],
        dtype=tl.float32,
    )
    acc = tl.zeros([BLOCK_M_SIZE, HEAD_DIM], dtype=tl.float32)

    acc, d_i = _attn_fwd_inner(
        acc,
        m_i,
        d_i,
        q,
        k_ptrs,
        v_ptrs,
        k_seq_stride,
        v_seq_stride,
        offs_m,
        qk_scale,
        n_size,
        BLOCK_M_SIZE,
        BLOCK_N_SIZE,
        v_ptr.dtype.element_ty == tl.float8e5,
    )

    acc = acc / d_i[:, None]
    out_mask = offs_m[:, None] < m_size
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.no_grad()
def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk_scale):

    BLOCK_SIZE = 64
    num_kv_groups = q.shape[1] // k.shape[1]
    output = torch.empty_like(q)

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"

    bs, n_heads, m_size, head_dim = q.size()

    n_size = k.shape[2]

    grid = lambda meta: (triton.cdiv(m_size, BLOCK_SIZE), bs * n_heads, 1)

    flash_attention_v2_kernel[grid](
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
        head_dim,
        BLOCK_SIZE,
        BLOCK_SIZE,
        qk_scale,
    )
    return output
