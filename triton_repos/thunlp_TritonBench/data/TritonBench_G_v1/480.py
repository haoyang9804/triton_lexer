import math
import torch
import triton
import triton.language as tl

_BLOCK_N = 64
_BLOCK_M = 64


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    N_CTX,
    sliding_window_offset,
    sliding_window_size,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    COMPLEMENT_SLIDING_WINDOW: tl.constexpr,
):

    if SLIDING_WINDOW and not COMPLEMENT_SLIDING_WINDOW:
        if COMPLEMENT_SLIDING_WINDOW:
            lo = 0
            hi = (
                (
                    (start_m + 1) * BLOCK_M
                    + sliding_window_offset
                    - sliding_window_size
                    + BLOCK_N
                    - 1
                )
                // BLOCK_N
            ) * BLOCK_N
        else:
            lo = (
                (start_m * BLOCK_M + sliding_window_offset - sliding_window_size + 1)
                // BLOCK_N
            ) * BLOCK_N
            hi = (
                (((start_m + 1) * BLOCK_M - 1) + sliding_window_offset + BLOCK_N)
                // BLOCK_N
            ) * BLOCK_N
            if lo < 0:
                lo = 0
            if hi > N_CTX:
                hi = N_CTX

            lo = tl.multiple_of(lo, BLOCK_N)
            K_block_ptr = tl.advance(K_block_ptr, (0, lo))
            V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    else:
        lo, hi = 0, N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if IS_EVEN_N:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale

        if SLIDING_WINDOW:
            dist = (
                tl.arange(0, BLOCK_M)[:, None]
                - tl.arange(0, BLOCK_N)[None, :]
                + start_m * BLOCK_M
                - start_n
                + sliding_window_offset
            )

            if COMPLEMENT_SLIDING_WINDOW:
                mask = dist >= sliding_window_size
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)

            qk = tl.where(mask, qk, float("-inf"))

        if not IS_EVEN_N:
            qk = tl.where(
                ((tl.arange(0, BLOCK_N) + start_n) < N_CTX)[None, :], qk, float("-inf")
            )

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)

        if not IS_EVEN_N:
            p = tl.where(((tl.arange(0, BLOCK_N) + start_n) < N_CTX)[None, :], p, 0)

        l_ij = tl.sum(p, 1)

        tmp = m_i - m_ij
        alpha_mask = tmp != tmp
        alpha = tl.math.exp2(tmp)
        alpha = tl.where(alpha_mask, 1.0, alpha)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        if IS_EVEN_N:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.heuristics(
    {
        "IS_EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "IS_EVEN_N": lambda args: args["NKV_CTX"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    L,
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
    Z,
    H,
    H_KV,
    N_CTX,
    ROUND_CTX,
    NKV_CTX,
    sliding_window_offset,
    sliding_window_size,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    END: tl.constexpr,
    INIT: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    COMPLEMENT_SLIDING_WINDOW: tl.constexpr,
):

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H // H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_hkv.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(NKV_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, NKV_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(ROUND_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    m_ptrs = M + off_hz * ROUND_CTX + offs_m
    l_ptrs = L + off_hz * ROUND_CTX + offs_m
    if INIT:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    else:

        m_i = tl.load(m_ptrs).to(tl.float32)
        l_i = tl.load(l_ptrs).to(tl.float32)
        acc = tl.load(O_block_ptr).to(tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634

    if IS_EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,
        start_m,
        qk_scale,
        NKV_CTX,
        sliding_window_offset,
        sliding_window_size,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        SLIDING_WINDOW,
        IS_EVEN_M,
        IS_EVEN_N,
        COMPLEMENT_SLIDING_WINDOW,
    )

    if END:
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
    else:
        tl.store(l_ptrs, l_i)

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def _forward(
    q,
    k,
    v,
    sm_scale,
    o=None,
    m=None,
    l=None,
    end=False,
    sliding_window=None,
    init=False,
    complement_sliding_window=False,
):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    q_round_len = math.ceil(q.shape[2] / 64) * 64

    if sliding_window is not None:
        sliding_window_offset, sliding_window_size = sliding_window
    else:
        sliding_window_offset, sliding_window_size = None, None

    grid = lambda META: (
        triton.cdiv(q.shape[2], META["BLOCK_M"]),
        q.shape[0] * q.shape[1],
    )

    global _BLOCK_N
    global _BLOCK_M

    try:
        with torch.cuda.device(q.device):
            _attn_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                m,
                o,
                l,
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
                k.shape[1],
                q.shape[2],
                q_round_len,
                k.shape[2],
                sliding_window_offset,
                sliding_window_size,
                BLOCK_DMODEL=Lk,
                END=end,
                INIT=init,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                SLIDING_WINDOW=(sliding_window is not None),
                COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
                num_warps=4,
                num_stages=4,
            )
    except triton.OutOfResources as E:
        _BLOCK_N = _BLOCK_N // 2
        _BLOCK_M = _BLOCK_M // 2
        from warnings import warn

        warn(
            f"Triton Attention Output Resources. {E}\nUse smaller block size {_BLOCK_N}."
        )
        with torch.cuda.device(q.device):
            _attn_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                m,
                o,
                l,
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
                k.shape[1],
                q.shape[2],
                q_round_len,
                k.shape[2],
                sliding_window_offset,
                sliding_window_size,
                BLOCK_DMODEL=Lk,
                END=end,
                INIT=init,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                SLIDING_WINDOW=(sliding_window is not None),
                COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
                num_warps=4,
                num_stages=4,
            )

    if end:
        o = o[:, :, : q.shape[2], :].contiguous().to(q.dtype)

    return o, m, l


import torch
import math


def test_forward():

    batch_size = 2
    num_heads = 4
    seq_len = 128
    d_model = 64

    q = torch.randn(
        (batch_size, num_heads, seq_len, d_model), device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        (batch_size, num_heads, seq_len, d_model), device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        (batch_size, num_heads, seq_len, d_model), device="cuda", dtype=torch.float16
    )

    o = torch.zeros_like(q)
    m = torch.zeros(
        (batch_size, num_heads, seq_len), device="cuda", dtype=torch.float32
    )
    l = torch.zeros(
        (batch_size, num_heads, seq_len), device="cuda", dtype=torch.float32
    )

    sm_scale = 1.0 / math.sqrt(d_model)

    sliding_window = (0, 64)
    complement_sliding_window = False
    o1, m1, l1 = _forward(
        q,
        k,
        v,
        sm_scale,
        o,
        m,
        l,
        end=True,
        sliding_window=sliding_window,
        init=True,
        complement_sliding_window=complement_sliding_window,
    )

    complement_sliding_window = True
    o2, m2, l2 = _forward(
        q,
        k,
        v,
        sm_scale,
        o,
        m,
        l,
        end=True,
        sliding_window=sliding_window,
        init=True,
        complement_sliding_window=complement_sliding_window,
    )

    sliding_window = None
    o3, m3, l3 = _forward(
        q,
        k,
        v,
        sm_scale,
        o,
        m,
        l,
        end=True,
        sliding_window=sliding_window,
        init=True,
        complement_sliding_window=False,
    )

    sliding_window = (0, 64)
    complement_sliding_window = False
    o4, m4, l4 = _forward(
        q,
        k,
        v,
        sm_scale,
        o,
        m,
        l,
        end=True,
        sliding_window=sliding_window,
        init=False,
        complement_sliding_window=complement_sliding_window,
    )

    return {
        "test_case_1": (o1, m1, l1),
        "test_case_2": (o2, m2, l2),
        "test_case_3": (o3, m3, l3),
        "test_case_4": (o4, m4, l4),
    }


result_gold = test_forward()
