import triton
import triton.language as tl
from dropout import fast_dropout_mask
from masked_load_store import load_fn
from triton.language.extra import libdevice
from composed_tensors import (
    composed_offs_1d,
    composed_load_with_offset,
    composed_dot_both,
    composed_dot_rhs,
    composed_mul_lhs,
    composed_mul_acc,
)


@triton.jit
def dot(BLOCK_M: tl.constexpr, QDIM: tl.constexpr, KDIM: tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)


@triton.jit
def bwd_inner_dk_dv(
    dk0,
    dk1,
    dk2,
    dv0,
    dv1,
    dv2,
    qk_scale,
    bias_scale,
    q_ptrs0,
    q_ptrs1,
    q_ptrs2,
    q_stride,
    kt0,
    kt1,
    kt2,
    vt0,
    vt1,
    vt2,
    B_ptr,
    stride_bm,
    stride_bn,
    do_ptrs0,
    do_ptrs1,
    do_ptrs2,
    do_stride,
    l_ptrs,
    D_ptrs,
    seqlen_q,
    seqlen_k,
    head_dim,
    start_k,
    nblocks_1,
    nblocks_2,
    Block_range_1,
    Block_range_2,
    idropout_p,
    dropout_scale,
    philox_seed,
    batch_philox_offset,
    philox_offset_stride,
    window_left,
    window_right,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL0,
    BLOCK_DMODEL1,
    BLOCK_DMODEL2,
    BLOCK_N: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):

    offs_k = start_k + tl.arange(0, BLOCK_N)
    offs_q = tl.arange(0, BLOCK_M)

    for block_index in range(nblocks_1 + nblocks_2):

        if Block_range_2 is None:
            start_qi = block_index + Block_range_1
        else:
            start_qi = (
                block_index + Block_range_1
                if block_index < nblocks_1
                else (block_index - nblocks_1 + Block_range_2)
            )
        start_q = start_qi * BLOCK_M

        offs_q_curr = offs_q[:, None] + start_q

        if FULL_BLOCKS and not PADDED_HEAD:
            q_offs_m = None
        else:
            q_offs_m = start_q + tl.arange(0, BLOCK_M)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        PADDED_SEQ: tl.constexpr = not FULL_BLOCKS
        q0, q1, q2 = composed_load_with_offset(
            q_ptrs0,
            q_ptrs1,
            q_ptrs2,
            start_q,
            q_stride,
            offs_q,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            seqlen_q,
            head_dim,
            other=0.0,
            PADDED_ROW=PADDED_SEQ,
            PADDED_COL=PADDED_HEAD,
            TRANSPOSED=False,
        )

        if not FULL_BLOCKS or IS_CAUSAL:
            mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
            MS = offs_q + start_q
            NS = offs_k
            if not FULL_BLOCKS:
                q_mask = MS[:, None] < seqlen_q
                mask = mask & q_mask

            if start_k + BLOCK_N > seqlen_k:
                k_mask = NS[None, :] < seqlen_k
                mask = mask & k_mask

            if IS_CAUSAL:
                right_mask = MS[:, None] + window_right >= NS[None, :]
                mask = mask & right_mask
                left_mask = MS[:, None] - window_left <= NS[None, :]
                mask = mask & left_mask

            qk = tl.where(mask, qk, float("-inf"))

        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            bias_ptrs = B_ptr + offs_q_curr * stride_bm + offs_k[None, :] * stride_bn

            if not FULL_BLOCKS:
                mask = (offs_q_curr < seqlen_q) & (offs_k < seqlen_k)[None, :]
                bias = tl.load(bias_ptrs, mask=mask, other=0.0)

            else:
                bias = tl.load(bias_ptrs)
            qk += bias * bias_scale
        else:
            tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")

        qk = composed_dot_both(
            q0, q1, q2, kt0, kt1, kt2, qk, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2
        )

        if FULL_BLOCKS:
            Di = tl.load(D_ptrs + offs_q_curr)
            l_i = tl.load(l_ptrs + offs_q_curr)
        else:
            d_lse_ptrs_mask = offs_q_curr < seqlen_q
            Di = tl.load(D_ptrs + offs_q_curr, mask=d_lse_ptrs_mask, other=0.0)
            l_i = tl.load(l_ptrs + offs_q_curr, mask=d_lse_ptrs_mask, other=0.0)

        p = tl.math.exp2(qk_scale * qk - l_i)

        if not FULL_BLOCKS or IS_CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)

        if ENABLE_DROPOUT:

            keep = fast_dropout_mask(
                philox_seed,
                idropout_p,
                batch_philox_offset,
                start_q,
                start_k,
                BLOCK_M,
                BLOCK_N,
                philox_offset_stride,
            )

            p_dropped = tl.where(keep, p * dropout_scale, 0.0)
        else:
            p_dropped = p

        do0, do1, do2 = composed_load_with_offset(
            do_ptrs0,
            do_ptrs1,
            do_ptrs2,
            start_q,
            do_stride,
            offs_q,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            seqlen_q,
            head_dim,
            other=0.0,
            PADDED_ROW=PADDED_SEQ,
            PADDED_COL=PADDED_HEAD,
            TRANSPOSED=False,
        )

        if BLOCK_M == 1:

            dv0, dv1, dv2 = composed_mul_acc(
                do0,
                do1,
                do2,
                p_dropped.to(do0.dtype),
                dv0,
                dv1,
                dv2,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
            )
        else:

            dv0, dv1, dv2 = composed_dot_rhs(
                p_dropped,
                do0,
                do1,
                do2,
                dv0,
                dv1,
                dv2,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                TRANSPOSE_LHS=True,
            )

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        dp += composed_dot_both(
            do0,
            do1,
            do2,
            vt0,
            vt1,
            vt2,
            dp,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
        )
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp * dropout_scale, 0)

        ds = p * (dp - Di)

        if BLOCK_M == 1:
            dk0, dk1, dk2 = composed_mul_acc(
                q0,
                q1,
                q2,
                ds.to(q_ptrs0.dtype.element_ty),
                dk0,
                dk1,
                dk2,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
            )

        else:

            dk0, dk1, dk2 = composed_dot_rhs(
                ds,
                q0,
                q1,
                q2,
                dk0,
                dk1,
                dk2,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                TRANSPOSE_LHS=True,
            )

    return dk0, dk1, dk2, dv0, dv1, dv2
