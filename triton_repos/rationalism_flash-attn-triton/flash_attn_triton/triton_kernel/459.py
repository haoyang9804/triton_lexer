import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

DEVICE = "cuda"


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
    dropout_p,
    dropout_seed,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    IS_BF16: tl.constexpr,
):

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)

    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    dropout_scale = 1.0 / (1.0 - dropout_p) if USE_DROPOUT else 1.0

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)

        if USE_DROPOUT:

            row_offsets = offs_m[:, None] + start_m * BLOCK_M
            col_offsets = start_n + offs_n[None, :]

            combined_offsets = row_offsets * N_CTX + col_offsets

            rng = tl.rand(dropout_seed, combined_offsets)
            dropout_mask = rng > dropout_p

            p = tl.where(dropout_mask, p / (1.0 - dropout_p), 0.0)

        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        if IS_BF16:
            p = p.to(tl.bfloat16)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [32, 64, 128]
    for BN in [16, 32, 64]
    for s in ([3, 4, 5, 7])
    for w in [4, 8]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    if BLOCK_M < BLOCK_N:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    dropout_p,
    dropout_seed,
    M,
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
    Z,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q = tl.load(Q_block_ptr)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            qk_scale,
            dropout_p,
            dropout_seed,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            USE_DROPOUT,
            IS_BF16,
        )

    if STAGE & 2:

        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            qk_scale,
            dropout_p,
            dropout_seed,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            USE_DROPOUT,
            IS_BF16,
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):

    start_idx = tl.program_id(0) * BLOCK_M

    off_m = start_idx + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)

    mask = off_m < N_CTX

    off_m_valid = tl.where(mask, off_m, N_CTX - 1)

    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m_valid[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO
        + off_hz * HEAD_DIM * N_CTX
        + off_m_valid[:, None] * HEAD_DIM
        + off_n[None, :]
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    tl.store(Delta + off_hz * N_CTX + off_m_valid, delta, mask=mask)


@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    Q,
    k,
    v,
    sm_scale,
    DO,
    M,
    D,
    dropout_p,
    dropout_seed,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    start_m,
    num_steps,
    MASK: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    dropout_scale = 1.0 / (1.0 - dropout_p) if USE_DROPOUT else 1.0
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)

        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])

        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)

        if USE_DROPOUT:

            row_offsets = offs_m[None, :]
            col_offsets = offs_n[:, None]
            combined_offsets = row_offsets * N_CTX + col_offsets

            rng = tl.rand(dropout_seed, combined_offsets)
            dropout_mask = rng > dropout_p

            pT = tl.where(dropout_mask, pT * dropout_scale, 0.0)

        do = tl.load(do_ptrs)

        ppT = pT
        if IS_BF16:
            ppT = ppT.to(tl.bfloat16)
        else:
            ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)

        Di = tl.load(D + offs_m)

        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        if IS_BF16:
            dsT = dsT.to(tl.bfloat16)
        else:
            dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))

        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,
    do,
    m,
    D,
    dropout_p,
    dropout_seed,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m,
    start_n,
    num_steps,
    MASK: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

    Di = tl.load(D + offs_m)

    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    dropout_scale = 1.0 / (1.0 - dropout_p) if USE_DROPOUT else 1.0
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)

        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)

        if USE_DROPOUT:

            row_offsets = offs_m[:, None]
            col_offsets = curr_n + tl.arange(0, BLOCK_N2)[None, :]
            combined_offsets = row_offsets * N_CTX + col_offsets

            rng = tl.rand(dropout_seed, combined_offsets)
            dropout_mask = rng > dropout_p

            p = tl.where(dropout_mask, p * dropout_scale, 0.0)

        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        if IS_BF16:
            ds = ds.to(tl.bfloat16)
        else:
            ds = ds.to(tl.float16)

        dq += tl.dot(ds, tl.trans(kT))

        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    dropout_p,
    dropout_seed,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    CAUSAL: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:

        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,
            Q,
            k,
            v,
            sm_scale,
            DO,
            M,
            D,
            dropout_p,
            dropout_seed,
            stride_tok,
            stride_d,
            H,
            N_CTX,
            MASK_BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,
            start_n,
            start_m,
            num_steps,
            MASK=True,
            USE_DROPOUT=USE_DROPOUT,
            IS_BF16=IS_BF16,
        )
        start_m += num_steps * MASK_BLOCK_M1
        num_steps = (N_CTX - start_m) // BLOCK_M1

        if num_steps > 0:
            dk, dv = _attn_bwd_dkdv(
                dk,
                dv,
                Q,
                k,
                v,
                sm_scale,
                DO,
                M,
                D,
                dropout_p,
                dropout_seed,
                stride_tok,
                stride_d,
                H,
                N_CTX,
                BLOCK_M1,
                BLOCK_N1,
                HEAD_DIM,
                start_n,
                start_m,
                num_steps,
                MASK=False,
                USE_DROPOUT=USE_DROPOUT,
                IS_BF16=IS_BF16,
            )
    else:

        num_steps = N_CTX // BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,
            Q,
            k,
            v,
            sm_scale,
            DO,
            M,
            D,
            dropout_p,
            dropout_seed,
            stride_tok,
            stride_d,
            H,
            N_CTX,
            BLOCK_M1,
            BLOCK_N1,
            HEAD_DIM,
            start_n,
            0,
            num_steps,
            MASK=False,
            USE_DROPOUT=USE_DROPOUT,
            IS_BF16=IS_BF16,
        )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:

        end_n = start_m + BLOCK_M2
        MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR

        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq,
            q,
            K,
            V,
            do,
            m,
            D,
            dropout_p,
            dropout_seed,
            stride_tok,
            stride_d,
            H,
            N_CTX,
            BLOCK_M2,
            MASK_BLOCK_N2,
            HEAD_DIM,
            start_m,
            end_n - num_steps * MASK_BLOCK_N2,
            num_steps,
            MASK=True,
            USE_DROPOUT=USE_DROPOUT,
            IS_BF16=IS_BF16,
        )
        end_n -= num_steps * MASK_BLOCK_N2

        if end_n > 0:
            num_steps = end_n // BLOCK_N2
            dq = _attn_bwd_dq(
                dq,
                q,
                K,
                V,
                do,
                m,
                D,
                dropout_p,
                dropout_seed,
                stride_tok,
                stride_d,
                H,
                N_CTX,
                BLOCK_M2,
                BLOCK_N2,
                HEAD_DIM,
                start_m,
                0,
                num_steps,
                MASK=False,
                USE_DROPOUT=USE_DROPOUT,
                IS_BF16=IS_BF16,
            )
    else:

        num_steps = N_CTX // BLOCK_N2
        dq = _attn_bwd_dq(
            dq,
            q,
            K,
            V,
            do,
            m,
            D,
            dropout_p,
            dropout_seed,
            stride_tok,
            stride_d,
            H,
            N_CTX,
            BLOCK_M2,
            BLOCK_N2,
            HEAD_DIM,
            start_m,
            0,
            num_steps,
            MASK=False,
            USE_DROPOUT=USE_DROPOUT,
            IS_BF16=IS_BF16,
        )

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dropout_p, dropout_seed):

        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]

        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        grid = lambda args: (
            triton.cdiv(q.shape[2], args["BLOCK_M"]),
            q.shape[0] * q.shape[1],
            1,
        )
        ctx.grid = grid

        USE_DROPOUT = dropout_p > 0.0

        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            dropout_p,
            dropout_seed,
            M,
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
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
            USE_DROPOUT=USE_DROPOUT,
            IS_BF16=(True if q.dtype == torch.bfloat16 else False),
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.dropout_seed = dropout_seed
        ctx.use_dropout = USE_DROPOUT
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        if not do.is_contiguous():
            do = do.contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 16, 64, 64, 16
        BLK_SLICE_FACTOR = 1
        RCP_LN2 = 1.4426950408889634
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)

        pre_grid = (triton.cdiv(N_CTX, PRE_BLOCK), BATCH * N_HEAD)

        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do, delta, BATCH, N_HEAD, N_CTX, BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM
        )

        grid = (triton.cdiv(N_CTX, BLOCK_N1), 1, BATCH * N_HEAD)

        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,
            M,
            delta,
            ctx.dropout_p,
            ctx.dropout_seed,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            N_HEAD,
            N_CTX,
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=ctx.HEAD_DIM,
            USE_DROPOUT=ctx.use_dropout,
            CAUSAL=ctx.causal,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            IS_BF16=(True if q.dtype == torch.bfloat16 else False),
        )

        return dq, dk, dv, None, None, None, None


def attention(q, k, v, causal=False, sm_scale=None, dropout_p=0.0, dropout_seed=None):

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    if dropout_seed is None:
        dropout_seed = torch.randint(0, 2**31 - 1, (), device=q.device).item()

    return _attention.apply(q, k, v, causal, sm_scale, dropout_p, dropout_seed)


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dropout_p", [0.0])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dropout_p, dtype=torch.float16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5
    dropout_seed = 1337
    dout = torch.randn_like(q)

    torch.manual_seed(dropout_seed)
    dropout = torch.nn.Dropout(dropout_p)

    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()

    p_dropped = dropout(p)

    ref_out = torch.matmul(p_dropped, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    torch.manual_seed(20)

    tri_out = attention(q, k, v, causal, sm_scale, dropout_p, dropout_seed).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    atol = 1e-1 if dropout_p > 0 else 1e-2
    rtol = 1e-1 if dropout_p > 0 else 0.0

    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dv, tri_dv, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=atol, rtol=rtol)


HAS_FLASH = False
import math

BATCH, N_HEADS, HEAD_DIM = 4, 32, 64

configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        for dropout_p in [0.0]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=[
                        "triton-fp16-dropout" if dropout_p > 0 else "triton-fp16"
                    ],
                    line_names=[
                        "Triton [FP16 + Dropout]" if dropout_p > 0 else "Triton [FP16]"
                    ],
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="TFLOPS",
                    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-dropout={dropout_p}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                        "dropout_p": dropout_p,
                    },
                )
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, dropout_p=0.0, device=DEVICE
):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        k = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        v = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        sm_scale = 1.3
        dropout_seed = 1337 if dropout_p > 0 else None

        fn = lambda: attention(q, k, v, causal, sm_scale, dropout_p, dropout_seed)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, HEAD_DIM),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":

    test_op(1, 2, 1024, 64, True, 0.0)
    print("All tests passed!")

    bench_flash_attention.run(save_path=".", print_data=True)
