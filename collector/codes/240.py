import triton
import triton.language as tl
from dropout import fast_dropout_mask
from masked_load_store import load_fn, mstore2d
from triton.language.extra import libdevice
from composed_tensors import (
    composed_offs_1d,
    composed_load_with_offset,
    composed_dot_rhs,
    composed_mul_lhs,
)


IS_JIT_COMPILING = False

if IS_JIT_COMPILING:
    from triton.language import constexpr as constexpr_or_i32
    from triton.language import constexpr as constexpr_or_f32
    from triton.language import constexpr as constexpr_or_bool
else:
    from triton.language import int32 as constexpr_or_i32
    from triton.language import float32 as constexpr_or_f32
    from triton.language import int1 as constexpr_or_bool


@triton.jit
def _attn_fwd_inner(
    acc0,
    acc1,
    acc2,
    l_i,
    m_i,
    Qk_scale: constexpr_or_f32,
    q0,
    q1,
    q2,
    k_ptrs0,
    k_ptrs1,
    k_ptrs2,
    v_ptrs0,
    v_ptrs1,
    v_ptrs2,
    stride_kn,
    stride_vk,
    B_ptrs,
    stride_bn,
    start_M,
    nblocks_1,
    nblocks_2,
    Block_range_1,
    Block_range_2,
    actual_seqlen_k,
    actual_seqlen_q,
    Head_dim,
    idropout_p,
    philox_seed,
    batch_philox_offset,
    philox_offset_stride,
    encoded_sm_base,
    Max_seqlen_k,
    window_left,
    window_right,
    alibi_slope,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL0: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
    BLOCK_DMODEL2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    INT8_GEMM: tl.constexpr,
    INT8_KV: tl.constexpr,
    USE_P_SCALE: tl.constexpr,
):

    for block_index in range(nblocks_1 + nblocks_2):

        if Block_range_2 is None:
            start_n = block_index + Block_range_1
        else:
            start_n = (
                block_index + Block_range_1
                if block_index < nblocks_1
                else (block_index - nblocks_1 + Block_range_2)
            )
        start_N = start_n * BLOCK_N

        k0, k1, k2 = composed_load_with_offset(
            k_ptrs0,
            k_ptrs1,
            k_ptrs2,
            start_N,
            stride_kn,
            OFFS_N,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            actual_seqlen_k,
            Head_dim,
            other=0.0,
            PADDED_ROW=MASK_STEPS,
            PADDED_COL=PADDED_HEAD,
            TRANSPOSED=True,
        )
        if PRE_LOAD_V:

            v0, v1, v2 = composed_load_with_offset(
                v_ptrs0,
                v_ptrs1,
                v_ptrs2,
                start_N,
                stride_kn,
                OFFS_N,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                actual_seqlen_k,
                Head_dim,
                other=0.0,
                PADDED_ROW=MASK_STEPS,
                PADDED_COL=PADDED_HEAD,
                TRANSPOSED=False,
            )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if MASK_STEPS or IS_CAUSAL:

            mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
            MS = OFFS_M
            NS = start_N + OFFS_N

            if MASK_STEPS:
                q_mask = MS[:, None] < actual_seqlen_q
                mask = mask & q_mask

            if start_N + BLOCK_N > actual_seqlen_k:
                k_mask = NS[None, :] < actual_seqlen_k
                mask = mask & k_mask

            if IS_CAUSAL:
                right_mask = MS[:, None] + window_right >= NS[None, :]
                mask = mask & right_mask
                left_mask = MS[:, None] - window_left <= NS[None, :]
                mask = mask & left_mask

            qk = tl.where(mask, qk, float("-inf"))

        if INT8_GEMM:
            qk += (((tl.dot(q, k).to(tl.float32) * q_descale)) * k_descale) * Qk_scale
        else:
            if INT8_KV:
                k = (k * k_descale).to(q.type.element_ty)

            qk += Qk_scale * tl.dot(q0, k0)
            if BLOCK_DMODEL1 > 0:
                qk += Qk_scale * tl.dot(q1, k1)
            if BLOCK_DMODEL2 > 0:
                qk += Qk_scale * tl.dot(q2, k2)

        if B_ptrs is not None:
            NS = start_N + OFFS_N
            bias_ptr = B_ptrs + NS[None, :] * stride_bn

            if MASK_STEPS:
                mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                    NS[None, :] < actual_seqlen_k
                )
                bias = tl.load(bias_ptr, mask=mask, other=0.0)

            else:
                bias = tl.load(bias_ptr)

            qk += bias * 1.44269504089

        if alibi_slope is not None:

            global_m_positions = start_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_N + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(
                alibi_slope,
                actual_seqlen_q,
                actual_seqlen_k,
                global_m_positions,
                global_n_positions,
            )
            qk += alibi_block * 1.44269504089

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        if MASK_STEPS or IS_CAUSAL:
            if Qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)

        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:

            keep = fast_dropout_mask(
                philox_seed,
                idropout_p,
                batch_philox_offset,
                start_M,
                start_N,
                BLOCK_M,
                BLOCK_N,
                philox_offset_stride,
            )
            if RETURN_ENCODED_SOFTMAX:
                mstore2d(
                    tl.where(keep, p, -p).to(encoded_sm_base.type.element_ty),
                    BLOCK_M,
                    BLOCK_N,
                    o_base=encoded_sm_base,
                    o_start_row=start_M,
                    o_start_col=start_N,
                    o_rows=actual_seqlen_q,
                    o_cols=actual_seqlen_k,
                    stride_row=Max_seqlen_k,
                    stride_col=1,
                )
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            mstore2d(
                p.to(encoded_sm_base.type.element_ty),
                BLOCK_M,
                BLOCK_N,
                o_base=encoded_sm_base,
                o_start_row=start_M,
                o_start_col=start_N,
                o_rows=actual_seqlen_q,
                o_cols=actual_seqlen_k,
                stride_row=Max_seqlen_k,
                stride_col=1,
            )

        alpha = tl.math.exp2(m_i - m_ij)
        acc0, acc1, acc2 = composed_mul_lhs(
            acc0,
            acc1,
            acc2,
            alpha[:, None],
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
        )
        if not PRE_LOAD_V:
            v0, v1, v2 = composed_load_with_offset(
                v_ptrs0,
                v_ptrs1,
                v_ptrs2,
                start_N,
                stride_kn,
                OFFS_N,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                actual_seqlen_k,
                Head_dim,
                other=0.0,
                PADDED_ROW=MASK_STEPS,
                PADDED_COL=PADDED_HEAD,
                TRANSPOSED=False,
            )

        l_i = l_i * alpha + l_ij

        m_i = m_ij

        if INT8_GEMM:
            if USE_P_SCALE:
                p = (p * p_scale).to(tl.int8)

                acc += tl.dot(p, v)
            else:

                acc += tl.dot(p, v.to(p.type.element_ty))
        else:
            if INT8_KV:
                v = (v * v_descale).to(p.type.element_ty)
            acc0, acc1, acc2 = composed_dot_rhs(
                p.to(v0.type.element_ty),
                v0,
                v1,
                v2,
                acc0,
                acc1,
                acc2,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
            )

    return acc0, acc1, acc2, l_i, m_i
