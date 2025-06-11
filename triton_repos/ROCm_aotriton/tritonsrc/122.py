import triton
import triton.language as tl
from bwd_inner_dq import bwd_inner_dq
from dropout import PHILOX_RN_PER_OFFSET
from masked_load_store import (
    load_fn,
    mstore2d,
    is_closed_interval_empty,
    parse_window,
    calculate_intervals,
    closed_interval_size,
)
from composed_tensors import (
    composed_offs_1d,
    composed_zeros_2d,
    composed_ptrs,
    composed_load,
    composed_advance,
    composed_to,
    composed_store,
    composed_mul_lhs,
    composed_mul_acc,
)


@triton.jit
def bwd_kernel_dq(
    Q,
    K,
    V,
    B,
    sm_scale,
    DO,
    DQ,
    DB,
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
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dbz,
    stride_dbh,
    stride_dbm,
    stride_dbn,
    num_head_q: "i32",
    num_head_k: "i32",
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,
    max_seqlen_q,
    max_seqlen_k,
    head_dim,
    dropout_p: tl.float32,
    philox_seed_ptr,
    philox_offset1: "*u64",
    philox_offset2: "u64",
    Window_left: "i32",
    Window_right: "i32",
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL_TYPE: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    tl.static_assert(BLOCK_DMODEL > 0, "BLOCK_DMODEL must be greater than 0")
    BLOCK_DMODEL_R0: tl.constexpr = BLOCK_DMODEL
    BLOCK_DMODEL0: tl.constexpr = 2 ** (BLOCK_DMODEL_R0.bit_length() - 1)
    BLOCK_DMODEL_R1: tl.constexpr = BLOCK_DMODEL_R0 - BLOCK_DMODEL0
    BLOCK_DMODEL1: tl.constexpr = (
        2 ** (BLOCK_DMODEL_R1.bit_length() - 1) if BLOCK_DMODEL_R1 > 0 else 0
    )
    BLOCK_DMODEL_R2: tl.constexpr = BLOCK_DMODEL_R1 - BLOCK_DMODEL1
    BLOCK_DMODEL2: tl.constexpr = (
        2 ** (BLOCK_DMODEL_R2.bit_length() - 1) if BLOCK_DMODEL_R2 > 0 else 0
    )
    BLOCK_DMODEL_R3: tl.constexpr = BLOCK_DMODEL_R2 - BLOCK_DMODEL2

    tl.static_assert(
        BLOCK_DMODEL_R3 == 0,
        f"BLOCK_DMODEL = {BLOCK_DMODEL} = 0b{BLOCK_DMODEL:b} cannot be factored into <= 3 power of two values",
    )
    tl.static_assert(
        BLOCK_DMODEL1 > 0 or BLOCK_DMODEL2 == 0, "Only trailing BLOCK_DMODELx can be 0"
    )
    IS_CAUSAL: tl.constexpr = CAUSAL_TYPE != 0

    philox_seed = 0
    philox_offset_base = philox_offset2
    philox_offset_stride = tl.cdiv(max_seqlen_k, PHILOX_RN_PER_OFFSET)
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
    start_q = tl.program_id(2) * BLOCK_M
    off_h_q = tl.program_id(1)
    off_h_k = (
        off_h_q if num_head_q == num_head_k else off_h_q // (num_head_q // num_head_k)
    )
    off_z = tl.program_id(0)
    num_z = tl.num_programs(2)
    off_zh = off_z * num_head_q + off_h_q * 1
    offs_q = start_q + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    cu_seqlens_q_start = 0
    cu_seqlens_k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    batch_index = off_z

    if num_seqlens > 0:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_q >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        batch_index = 0

    if num_seqlens < 0:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_q >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start

        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z

    q_ptrs0, q_ptrs1, q_ptrs2 = composed_ptrs(
        Q,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        batch_index,
        off_h_q,
        cu_seqlens_q_start + offs_q,
        BLOCK_DMODEL0,
        BLOCK_DMODEL1,
        BLOCK_DMODEL2,
    )
    if start_q + BLOCK_M <= seqlen_q:
        q0, q1, q2 = composed_load(
            q_ptrs0,
            q_ptrs1,
            q_ptrs2,
            offs_q,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            seqlen_q,
            head_dim,
            other=0.0,
            PADDED_ROW=False,
            PADDED_COL=PADDED_HEAD,
            TRANSPOSED=False,
        )
    else:
        q0, q1, q2 = composed_load(
            q_ptrs0,
            q_ptrs1,
            q_ptrs2,
            offs_q,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            seqlen_q,
            head_dim,
            other=0.0,
            PADDED_ROW=True,
            PADDED_COL=PADDED_HEAD,
            TRANSPOSED=False,
        )
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale
    kt_ptrs0, kt_ptrs1, kt_ptrs2 = composed_ptrs(
        K,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        batch_index,
        off_h_k,
        cu_seqlens_k_start + offs_n,
        BLOCK_DMODEL0,
        BLOCK_DMODEL1,
        BLOCK_DMODEL2,
        TRANSPOSED=True,
    )

    vt_ptrs0, vt_ptrs1, vt_ptrs2 = composed_ptrs(
        V,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        batch_index,
        off_h_k,
        cu_seqlens_k_start + offs_n,
        BLOCK_DMODEL0,
        BLOCK_DMODEL1,
        BLOCK_DMODEL2,
        TRANSPOSED=True,
    )

    do_ptrs0, do_ptrs1, do_ptrs2 = composed_ptrs(
        DO,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        batch_index,
        off_h_q,
        cu_seqlens_q_start + offs_q,
        BLOCK_DMODEL0,
        BLOCK_DMODEL1,
        BLOCK_DMODEL2,
    )

    if start_q + BLOCK_M <= seqlen_q:
        do0, do1, do2 = composed_load(
            do_ptrs0,
            do_ptrs1,
            do_ptrs2,
            offs_q,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            seqlen_q,
            head_dim,
            other=0.0,
            PADDED_ROW=False,
            PADDED_COL=PADDED_HEAD,
            TRANSPOSED=False,
        )
    else:
        do0, do1, do2 = composed_load(
            do_ptrs0,
            do_ptrs1,
            do_ptrs2,
            offs_q,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            seqlen_q,
            head_dim,
            other=0.0,
            PADDED_ROW=True,
            PADDED_COL=PADDED_HEAD,
            TRANSPOSED=False,
        )

    D_ptrs = D + off_zh * max_seqlen_q
    l_ptrs = L + off_zh * max_seqlen_q
    if ENABLE_DROPOUT:
        batch_philox_offset = (
            philox_offset_base + off_zh * max_seqlen_q * philox_offset_stride
        )
    else:
        batch_philox_offset = 0

    dq_offset = (
        batch_index * stride_dqz
        + off_h_q * stride_dqh
        + cu_seqlens_q_start * stride_dqm
    )
    DQ += dq_offset

    store_db = True
    if BIAS_TYPE == 0:
        B_ptr = 0
        DB_ptr = 0
    elif BIAS_TYPE == 1:

        B_ptr = B + off_h_q * stride_bh + batch_index * stride_bz
        if (stride_dbz == 0 and stride_dbh == 0) and stride_dbm == 0:
            store_db = False

        DB_ptr = DB + off_h_q * stride_dbh + batch_index * stride_dbz
    else:
        tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")

    window_left, window_right = parse_window(
        IS_CAUSAL, CAUSAL_TYPE, Window_left, Window_right, seqlen_q, seqlen_k
    )
    mask_on_seq_q = start_q + BLOCK_M > seqlen_q
    lb_lo, lb_hi, fb_lo, fb_hi, rb_lo, rb_hi = calculate_intervals(
        IS_CAUSAL,
        CAUSAL_TYPE,
        window_left,
        window_right,
        start_q,
        seqlen_q,
        seqlen_k,
        mask_on_seq_q,
        BLOCK_M,
        BLOCK_N,
    )
    lb_empty = is_closed_interval_empty(lb_lo, lb_hi)
    rb_empty = is_closed_interval_empty(rb_lo, rb_hi)
    fb_empty = is_closed_interval_empty(fb_lo, fb_hi)

    d_lse_ptrs_mask = offs_q < seqlen_q
    Di = tl.load(D_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)
    l_i = tl.load(l_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)

    idropout_p = ((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32) if ENABLE_DROPOUT else 0
    dropout_scale = 1.0 / (1.0 - dropout_p) if ENABLE_DROPOUT else 1.0
    dq0, dq1, dq2 = composed_zeros_2d(
        BLOCK_M, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2
    )
    if not fb_empty:
        nblocks_1 = closed_interval_size(fb_lo, fb_hi)
        dq0, dq1, dq2 = bwd_inner_dq(
            dq0,
            dq1,
            dq2,
            qk_scale,
            bias_scale,
            DB_ptr,
            store_db,
            q0,
            q1,
            q2,
            kt_ptrs0,
            kt_ptrs1,
            kt_ptrs2,
            stride_kn,
            vt_ptrs0,
            vt_ptrs1,
            vt_ptrs2,
            stride_vk,
            B_ptr,
            stride_bn,
            stride_bm,
            stride_dbn,
            stride_dbm,
            do0,
            do1,
            do2,
            Di,
            l_i,
            seqlen_q,
            seqlen_k,
            head_dim,
            start_q,
            nblocks_1,
            0,
            fb_lo,
            None,
            idropout_p,
            dropout_scale,
            philox_seed,
            batch_philox_offset,
            philox_offset_stride,
            window_left,
            window_right,
            BLOCK_M,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            BLOCK_N,
            True,
            False,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE,
        )

    if not (lb_empty and rb_empty):
        tl.debug_barrier()
        nblocks_1 = closed_interval_size(lb_lo, lb_hi)
        nblocks_2 = closed_interval_size(rb_lo, rb_hi)
        dq0, dq1, dq2 = bwd_inner_dq(
            dq0,
            dq1,
            dq2,
            qk_scale,
            bias_scale,
            DB_ptr,
            store_db,
            q0,
            q1,
            q2,
            kt_ptrs0,
            kt_ptrs1,
            kt_ptrs2,
            stride_kn,
            vt_ptrs0,
            vt_ptrs1,
            vt_ptrs2,
            stride_vk,
            B_ptr,
            stride_bn,
            stride_bm,
            stride_dbn,
            stride_dbm,
            do0,
            do1,
            do2,
            Di,
            l_i,
            seqlen_q,
            seqlen_k,
            head_dim,
            start_q,
            nblocks_1,
            nblocks_2,
            lb_lo,
            rb_lo,
            idropout_p,
            dropout_scale,
            philox_seed,
            batch_philox_offset,
            philox_offset_stride,
            window_left,
            window_right,
            BLOCK_M,
            BLOCK_DMODEL0,
            BLOCK_DMODEL1,
            BLOCK_DMODEL2,
            BLOCK_N,
            False,
            CAUSAL_TYPE,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE,
        )
    dq0, dq1, dq2 = composed_mul_lhs(
        dq0, dq1, dq2, sm_scale, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2
    )
    dq0, dq1, dq2 = composed_to(dq0, dq1, dq2, dq0.type.element_ty)
    composed_store(
        dq0,
        dq1,
        dq2,
        BLOCK_M,
        BLOCK_DMODEL0,
        BLOCK_DMODEL1,
        BLOCK_DMODEL2,
        o_base=DQ,
        o_start_row=start_q,
        o_start_col=0,
        o_rows=seqlen_q,
        o_cols=head_dim,
        stride_row=stride_dqm,
        stride_col=stride_dqk,
    )
