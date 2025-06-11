import triton
import triton.language as tl
from composed_tensors import (
    composed_ptrs,
    composed_load,
    composed_inner_product_fp32,
)


@triton.jit
def bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    seqlen_q,
    head_dim,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):

    tl.static_assert(D_HEAD > 0, "D_HEAD must be greater than 0")
    D_HEAD_R0: tl.constexpr = D_HEAD
    D_HEAD0: tl.constexpr = 2 ** (D_HEAD_R0.bit_length() - 1)
    D_HEAD_R1: tl.constexpr = D_HEAD_R0 - D_HEAD0
    D_HEAD1: tl.constexpr = 2 ** (D_HEAD_R1.bit_length() - 1) if D_HEAD_R1 > 0 else 0
    D_HEAD_R2: tl.constexpr = D_HEAD_R1 - D_HEAD1
    D_HEAD2: tl.constexpr = 2 ** (D_HEAD_R2.bit_length() - 1) if D_HEAD_R2 > 0 else 0
    D_HEAD_R3: tl.constexpr = D_HEAD_R2 - D_HEAD2

    tl.static_assert(
        D_HEAD_R3 == 0,
        f"D_HEAD = {D_HEAD} = 0b{D_HEAD:b} cannot be factored into <= 3 power of two values",
    )
    tl.static_assert(D_HEAD1 > 0 or D_HEAD2 == 0, "Only trailing D_HEADx can be 0")

    off_m = tl.program_id(2) * BLOCK_M
    offs_m = off_m + tl.arange(0, BLOCK_M)
    off_h = tl.program_id(1)
    off_z = tl.program_id(0)
    num_h = tl.num_programs(1)

    o_ptrs0, o_ptrs1, o_ptrs2 = composed_ptrs(
        Out,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        off_z,
        off_h,
        offs_m,
        D_HEAD0,
        D_HEAD1,
        D_HEAD2,
    )

    do_ptrs0, do_ptrs1, do_ptrs2 = composed_ptrs(
        DO,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_don,
        off_z,
        off_h,
        offs_m,
        D_HEAD0,
        D_HEAD1,
        D_HEAD2,
    )

    o0, o1, o2 = composed_load(
        o_ptrs0,
        o_ptrs1,
        o_ptrs2,
        offs_m,
        D_HEAD0,
        D_HEAD1,
        D_HEAD2,
        seqlen_q,
        head_dim,
        other=0.0,
        PADDED_ROW=True,
        PADDED_COL=PADDED_HEAD,
        TRANSPOSED=False,
    )
    do0, do1, do2 = composed_load(
        do_ptrs0,
        do_ptrs1,
        do_ptrs2,
        offs_m,
        D_HEAD0,
        D_HEAD1,
        D_HEAD2,
        seqlen_q,
        head_dim,
        other=0.0,
        PADDED_ROW=True,
        PADDED_COL=PADDED_HEAD,
        TRANSPOSED=False,
    )

    delta = composed_inner_product_fp32(
        o0, o1, o2, do0, do1, do2, D_HEAD0, D_HEAD1, D_HEAD2, axis=1
    )

    off_zh = off_z * num_h + off_h * 1

    delta_ptrs = Delta + off_zh * seqlen_q + off_m + tl.arange(0, BLOCK_M)
    overflow = off_m + BLOCK_M - seqlen_q
    if overflow > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow, dtype=tl.int32)
        mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(delta_ptrs, delta, mask=mask)
    else:
        tl.store(delta_ptrs, delta)


@triton.jit
def bwd_preprocess_varlen(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    cu_seqlens_q,
    max_seqlen_q,
    head_dim,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):

    tl.static_assert(D_HEAD > 0, "D_HEAD must be greater than 0")
    D_HEAD_R0: tl.constexpr = D_HEAD
    D_HEAD0: tl.constexpr = 2 ** (D_HEAD_R0.bit_length() - 1)
    D_HEAD_R1: tl.constexpr = D_HEAD_R0 - D_HEAD0
    D_HEAD1: tl.constexpr = 2 ** (D_HEAD_R1.bit_length() - 1) if D_HEAD_R1 > 0 else 0
    D_HEAD_R2: tl.constexpr = D_HEAD_R1 - D_HEAD1
    D_HEAD2: tl.constexpr = 2 ** (D_HEAD_R2.bit_length() - 1) if D_HEAD_R2 > 0 else 0
    D_HEAD_R3: tl.constexpr = D_HEAD_R2 - D_HEAD2

    tl.static_assert(
        D_HEAD_R3 == 0,
        f"D_HEAD = {D_HEAD} = 0b{D_HEAD:b} cannot be factored into <= 3 power of two values",
    )
    tl.static_assert(D_HEAD1 > 0 or D_HEAD2 == 0, "Only trailing D_HEADx can be 0")

    off_m = tl.program_id(2) * BLOCK_M
    offs_m = off_m + tl.arange(0, BLOCK_M)
    off_h = tl.program_id(1)
    off_z = tl.program_id(0)
    num_h = tl.num_programs(1)
    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
    seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
    if off_m >= seqlen_q:
        return

    o_ptrs0, o_ptrs1, o_ptrs2 = composed_ptrs(
        Out,
        stride_oh,
        stride_om,
        stride_om,
        stride_on,
        off_h,
        cu_seqlens_q_start,
        offs_m,
        D_HEAD0,
        D_HEAD1,
        D_HEAD2,
    )

    do_ptrs0, do_ptrs1, do_ptrs2 = composed_ptrs(
        DO,
        stride_doh,
        stride_dom,
        stride_dom,
        stride_don,
        off_h,
        cu_seqlens_q_start,
        offs_m,
        D_HEAD0,
        D_HEAD1,
        D_HEAD2,
    )

    o0, o1, o2 = composed_load(
        o_ptrs0,
        o_ptrs1,
        o_ptrs2,
        offs_m,
        D_HEAD0,
        D_HEAD1,
        D_HEAD2,
        seqlen_q,
        head_dim,
        other=0.0,
        PADDED_ROW=True,
        PADDED_COL=PADDED_HEAD,
        TRANSPOSED=False,
    )
    do0, do1, do2 = composed_load(
        do_ptrs0,
        do_ptrs1,
        do_ptrs2,
        offs_m,
        D_HEAD0,
        D_HEAD1,
        D_HEAD2,
        seqlen_q,
        head_dim,
        other=0.0,
        PADDED_ROW=True,
        PADDED_COL=PADDED_HEAD,
        TRANSPOSED=False,
    )

    delta = composed_inner_product_fp32(
        o0, o1, o2, do0, do1, do2, D_HEAD0, D_HEAD1, D_HEAD2, axis=1
    )

    off_zh = off_z * num_h + off_h * 1

    delta_ptrs = Delta + off_zh * max_seqlen_q + off_m + tl.arange(0, BLOCK_M)
    overflow = off_m + BLOCK_M - seqlen_q
    if overflow > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow, dtype=tl.int32)
        mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(delta_ptrs, delta, mask=mask)
    else:
        tl.store(delta_ptrs, delta)
