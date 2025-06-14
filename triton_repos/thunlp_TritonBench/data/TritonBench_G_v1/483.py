import torch
import triton
import triton.language as tl
import math


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
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
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    N_HEAD,
    H,
    N_CTX,
    start_position,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    start_m = tl.program_id(0)

    head_idx = tl.program_id(1)
    batch_id = head_idx // N_HEAD
    off_hz = head_idx % N_HEAD

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = (
        batch_id * stride_qz
        + off_hz * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    off_k = (
        batch_id * stride_kz
        + off_hz * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kk
    )
    off_v = (
        batch_id * stride_vz
        + off_hz * stride_vh
        + offs_n[:, None] * stride_vk
        + offs_d[None, :] * stride_vn
    )

    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, offs_m[:, None] < H, other=0.0)

    block_n_end = N_CTX
    if IS_CAUSAL:

        block_n_end = (start_m + 1) * BLOCK_N + start_position
    for start_n in range(0, block_n_end, BLOCK_N):
        block_n_offs = start_n + offs_n

        k = tl.load(k_ptrs, block_n_offs[:, None] < N_CTX, 0.0)
        if USE_FP8:
            k = k.to(tl.float8e5, bitcast=True)
            k = k.to(tl.float16)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))
        qk *= sm_scale
        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (block_n_offs[None, :] + start_position),
                qk,
                float("-inf"),
            )

        m_curr = tl.maximum(tl.max(qk, 1), m_prev)

        l_prev *= tl.exp(m_prev - m_curr)

        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev

        l_rcp = 1.0 / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]

        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs, block_n_offs[:, None] < N_CTX, 0.0)
        if USE_FP8:
            v = v.to(tl.float8e5, bitcast=True)
            v = v.to(tl.float16)
        acc += tl.dot(p, v)

        l_prev = l_curr
        m_prev = m_curr

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_o = (
        batch_id * stride_oz
        + off_hz * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, offs_m[:, None] < H)


def triton_fa(q, k, v, sm_scale, is_causal, start_position):
    assert q.dtype == torch.float16
    assert k.dtype == v.dtype and k.dtype in [torch.float16, torch.int8]

    BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    num_warps = 4 if Lk <= 64 else 8
    batch, head_size, m_size, dhead = q.size()
    grid = (triton.cdiv(m_size, BLOCK), head_size * batch)
    n_size = k.size(2)
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
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
        head_size,
        m_size,
        n_size,
        start_position=start_position,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk,
        USE_FP8=k.dtype == torch.int8,
        num_warps=num_warps,
        num_stages=2,
    )

    return o


import torch
import math


def test_triton_fa():

    xq = torch.randn([1, 16, 32, 128], dtype=torch.float16, device="cuda")
    keys = torch.randn([1, 16, 32, 128], dtype=torch.float16, device="cuda")
    values = torch.randn([1, 16, 32, 128], dtype=torch.float16, device="cuda")

    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    scale = 1 / math.sqrt(128)
    output_t1 = triton_fa(xq, keys, values, scale, False, 0)

    output_t2 = triton_fa(xq, keys, values, scale, True, 0)

    keys_int8 = keys.to(torch.int8)
    values_int8 = values.to(torch.int8)
    output_t3 = triton_fa(xq, keys_int8, values_int8, scale, False, 0)

    output_t4 = triton_fa(xq, keys_int8, values_int8, scale, True, 0)

    return {
        "test_case_1": output_t1,
        "test_case_2": output_t2,
        "test_case_3": output_t3,
        "test_case_4": output_t4,
    }


result_gold = test_triton_fa()
