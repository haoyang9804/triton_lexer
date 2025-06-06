import triton
import triton.language as tl
from fwd_kernel import dropout_mask, dropout_rng, dropout_offsets


@triton.jit
def dot(BLOCK_M: tl.constexpr, QDIM: tl.constexpr, KDIM: tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)


@triton.jit
def bwd_kernel_dk_dv(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DK,
    DV,
    L,
    D,
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
    seqlen_q,
    seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_N
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)

    offs_m = start_m + tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_M)

    q_offset = off_h * stride_qh + off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_offset = off_h * stride_kh + off_z * stride_kz
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_offset = off_h * stride_vh + off_z * stride_vz
    VT_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    do_offset = q_offset
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    off_zh = off_z * num_h + off_h * 1

    D_ptrs = D + off_zh * seqlen_q
    l_ptrs = L + off_zh * seqlen_q
    qk_scale = sm_scale * 1.44269504

    k = tl.load(K_block_ptr)
    k = (k * qk_scale).to(K_block_ptr.type.element_ty)
    vt = tl.load(VT_block_ptr)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    lo = (start_m // BLOCK_M) * BLOCK_M if CAUSAL else 0
    hi = seqlen_q
    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k

    for start_n in range(lo, hi, BLOCK_M):
        offs_m_curr = offs_n[:, None] + start_n

        q = tl.load(Q_block_ptr)
        do = tl.load(DO_block_ptr)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        qk += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, k)
        if CAUSAL:
            qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i)

        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_n * seqlen_k + start_m
            keep = dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k
            )

            if BLOCK_M == 1:
                dv += (
                    tl.where(keep, p / (1 - dropout_p), 0.0).to(Q.dtype.element_ty) * do
                )
            else:
                dv += tl.dot(
                    tl.where(tl.trans(keep), tl.trans(p) / (1 - dropout_p), 0.0).to(
                        Q.dtype.element_ty
                    ),
                    do,
                )
        else:
            if BLOCK_M == 1:
                dv += p.to(Q.dtype.element_ty) * do
            else:

                dv += tl.dot(tl.trans(p).to(do.dtype), do)

        Di = tl.load(D_ptrs + offs_m_curr)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        dp += tl.dot(do, vt)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1 - dropout_p), 0)

        ds = p * (dp - Di)

        if BLOCK_M == 1:
            dk += ds.to(Q.dtype.element_ty) * q
        else:

            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)

        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))

    DK_block_ptr = tl.make_block_ptr(
        base=DK + k_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + v_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(DK_block_ptr, (dk * sm_scale).to(DK.type.element_ty))
    tl.store(DV_block_ptr, dv.to(DV.type.element_ty))


@triton.jit
def bwd_kernel_dq(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    L,
    D,
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
    seqlen_q,
    seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q_offset = off_h * stride_qh + off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_offset = off_h * stride_kh + off_z * stride_kz
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_offset = off_h * stride_vh + off_z * stride_vz
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    off_zh = off_z * num_h + off_h * 1

    D_ptrs = D + off_zh * seqlen_q
    l_ptrs = L + off_zh * seqlen_q
    qk_scale = sm_scale * 1.44269504

    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
    do = tl.load(DO_block_ptr)
    Di = tl.load(D_ptrs + offs_m)
    l_i = tl.load(l_ptrs + offs_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    lo = 0
    hi = min(start_m + BLOCK_M, seqlen_k) if CAUSAL else seqlen_k
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k

    for start_n in range(lo, hi, BLOCK_N):

        kt = tl.load(K_block_ptr)
        vt = tl.load(V_block_ptr)

        qk = dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt)
        if CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float("-inf")
            )
        p = tl.math.exp2(qk - l_i[:, None])

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * seqlen_k + start_n
            keep = dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k
            )
            dp = tl.where(keep, dp / (1 - dropout_p), 0)

        ds = p * (dp - Di[:, None])

        if BLOCK_M == 1:
            dq += tl.view(kt, [BLOCK_DMODEL]) * ds.to(Q.type.element_ty)
        else:

            dq += tl.dot(ds.to(Q.type.element_ty), tl.trans(kt))

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))

    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(DQ_block_ptr, (dq * sm_scale).to(DQ_block_ptr.type.element_ty))
