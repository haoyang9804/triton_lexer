import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    sliding_window,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, -1e9)
        qk = tl.where(
            (start_n + offs_n[None, :]) > (offs_m[:, None] - sliding_window), qk, -1e9
        )

        m_ij = tl.max(qk, 1)

        m_ij = tl.where(m_ij == -1e9, 0.0, m_ij)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        l_i_new = tl.where(l_i_new == 0.0, 1e-9, l_i_new)

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return


@torch.no_grad()
def context_attention_fwd(
    q, k, v, o, b_start_loc, b_seq_len, max_input_len, sliding_window
):
    BLOCK = 128

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_group_num=kv_group_num,
        sliding_window=sliding_window,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test_context_attention_fwd():
    Z, H, N_CTX, D_HEAD = 4, 6, 1024, 128
    dtype = torch.float16
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.1, std=0.2
    )
    k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.4, std=0.2
    )
    v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.3, std=0.2
    )
    o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.3, std=0.2
    )

    max_input_len = N_CTX
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")

    b_seq_len[0] = 512
    b_seq_len[1] = 1024
    b_seq_len[2] = 512
    b_seq_len[3] = 1024

    for i in range(1, Z):
        b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]

    results = {}

    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, 10)
    results["test_case_1"] = o.clone()

    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, 20)
    results["test_case_2"] = o.clone()

    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len // 2, 10)
    results["test_case_3"] = o.clone()

    Z = 2
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.1, std=0.2
    )
    k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.4, std=0.2
    )
    v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.3, std=0.2
    )
    o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.3, std=0.2
    )
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_seq_len[0] = 512
    b_seq_len[1] = 1024
    for i in range(1, Z):
        b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, 10)
    results["test_case_4"] = o.clone()

    return results


result_gold = test_context_attention_fwd()
