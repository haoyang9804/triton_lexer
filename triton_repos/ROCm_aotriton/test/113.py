import triton
import triton.language as tl


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
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):

    off_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    num_h = tl.num_programs(1)
    o_offset = off_h * stride_oh + off_z * stride_oz
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(off_m, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )
    do_offset = off_h * stride_doh + off_z * stride_doz
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, D_HEAD),
        strides=(stride_dom, stride_don),
        offsets=(off_m, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )

    o = tl.load(O_block_ptr).to(tl.float32)
    do = tl.load(DO_block_ptr).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    off_zh = off_z * num_h + off_h * 1
    tl.store(Delta + off_zh * seqlen_q + off_m + tl.arange(0, BLOCK_M), delta)
