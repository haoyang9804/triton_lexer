import triton
import triton.language as tl


@triton.jit
def block_sparse_attention_kernel(
    out,
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    layout_csr_row_stride_h,
    layout_csr_col_stride_h,
    num_layout,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    num_heads,
    num_kv_heads,
    total_seq_len,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    tl.static_print(
        f"{BLOCK_M=} {BLOCK_N=} {BLOCK_D=} {EVEN_M=} {EVEN_N=} {NUM_D_BLOCKS=}"
    )

    q_seq_len = total_seq_len

    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    off_h = off_bh % num_heads
    off_b = off_bh // num_heads

    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups

    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h_kv * stride_kh
    V += off_b * stride_vb + off_h_kv * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :]
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :]

    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    if NUM_D_BLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if EVEN_M:
        q = tl.load(q_ptrs)
        if NUM_D_BLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_D)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len)
        if NUM_D_BLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_D, mask=offs_m[:, None] < q_seq_len)

    layout_h = off_h % num_layout

    layout_ptr = layout_csr_row_indices + layout_h * layout_csr_row_stride_h + start_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + 1).to(tl.int32)

    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(
            layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx
        ).to(tl.int32)
        start_n = col_idx * BLOCK_N

        if EVEN_N:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kn,
                mask=offs_n[None, :] + start_n < total_seq_len,
            )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if NUM_D_BLOCKS >= 2:
            if EVEN_N:
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_D)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn + BLOCK_D,
                    mask=offs_n[None, :] + start_n < total_seq_len,
                )
            qk += tl.dot(q2, k)

        qk *= softmax_scale

        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        if NUM_D_BLOCKS >= 2:
            acc2 = acc2 * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)

        if EVEN_N:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=offs_n[:, None] + start_n < total_seq_len,
            )
        acc += tl.dot(p, v)

        if NUM_D_BLOCKS >= 2:
            if EVEN_N:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_D)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn + BLOCK_D,
                    mask=offs_n[:, None] + start_n < total_seq_len,
                )
            acc2 += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

    off_o = (
        off_b * stride_ob
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :]
    )
    out_ptrs = out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
    if NUM_D_BLOCKS >= 2:
        tl.store(out_ptrs + BLOCK_D, acc2, mask=offs_m[:, None] < q_seq_len)


def block_sparse_attention(
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    layout_csr_row_stride_h,
    layout_csr_col_stride_h,
    num_layout,
    softmax_scale,
    num_heads,
    num_kv_heads,
    total_seq_len,
    BLOCK_M,
    EVEN_M,
    BLOCK_N,
    EVEN_N,
    BLOCK_D,
    NUM_D_BLOCKS,
):

    q_seq_len = total_seq_len
    grid = (triton.cdiv(q_seq_len, BLOCK_M), Q.shape[0] * num_heads)

    out = torch.empty(
        (Q.shape[0], num_heads, q_seq_len, Q.shape[-1]), device=Q.device, dtype=Q.dtype
    )

    block_sparse_attention_kernel[grid](
        out,
        Q,
        K,
        V,
        layout_csr_row_indices,
        layout_csr_col_indices,
        layout_csr_row_stride_h,
        layout_csr_col_stride_h,
        num_layout,
        softmax_scale,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        num_kv_heads,
        total_seq_len,
        BLOCK_M=BLOCK_M,
        EVEN_M=EVEN_M,
        BLOCK_N=BLOCK_N,
        EVEN_N=EVEN_N,
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
    )
    return out


import torch


def test_block_sparse_attention():

    B, H, M, D = 2, 4, 16, 32
    N = 32
    num_kv_heads = 2
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = 16
    NUM_D_BLOCKS = 2
    EVEN_M = True
    EVEN_N = True
    num_layout = 1
    softmax_scale = 1.0

    Q = torch.randn((B, H, M, D), device="cuda", dtype=torch.float32)
    K = torch.randn((B, num_kv_heads, N, D), device="cuda", dtype=torch.float32)
    V = torch.randn((B, num_kv_heads, N, D), device="cuda", dtype=torch.float32)

    layout_csr_row_indices = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    layout_csr_col_indices = torch.tensor(
        [0, 1, 2, 3], device="cuda", dtype=torch.int32
    )
    layout_csr_row_stride_h = 3
    layout_csr_col_stride_h = 4

    out1 = block_sparse_attention(
        Q,
        K,
        V,
        layout_csr_row_indices,
        layout_csr_col_indices,
        layout_csr_row_stride_h,
        layout_csr_col_stride_h,
        num_layout,
        softmax_scale,
        H,
        num_kv_heads,
        M,
        BLOCK_M,
        EVEN_M,
        BLOCK_N,
        EVEN_N,
        BLOCK_D,
        NUM_D_BLOCKS,
    )

    EVEN_M = False
    EVEN_N = False

    out2 = block_sparse_attention(
        Q,
        K,
        V,
        layout_csr_row_indices,
        layout_csr_col_indices,
        layout_csr_row_stride_h,
        layout_csr_col_stride_h,
        num_layout,
        softmax_scale,
        H,
        num_kv_heads,
        M,
        BLOCK_M,
        EVEN_M,
        BLOCK_N,
        EVEN_N,
        BLOCK_D,
        NUM_D_BLOCKS,
    )

    return {"test_case_1": out1, "test_case_2": out2}


result_gold = test_block_sparse_attention()
