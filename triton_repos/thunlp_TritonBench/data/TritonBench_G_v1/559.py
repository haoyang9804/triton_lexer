from typing import Literal
import torch
import triton
import triton.language as tl


KERNEL_META = dict()


def get_kernel_meta(tensor: torch.Tensor):

    return KERNEL_META


@triton.jit
def _div_up(val, other):
    return (val + other - 1) // other


@triton.jit
def _quant_int8(val):
    val_min = tl.min(val, 1)
    val_max = tl.max(val, 1)
    scales = (val_max - val_min) / 255
    zeros = -val_min / scales
    q_val = (val / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    return q_val, scales, zeros


@triton.jit
def _quant_int4(val1, val2):
    val1 = val1.to(tl.float32)
    val2 = val2.to(tl.float32)
    val_min = tl.min(tl.minimum(val1, val2), 1)
    val_max = tl.max(tl.maximum(val1, val2), 1)
    scales = (val_max - val_min) / 15
    zeros = -val_min / scales
    q_val1 = (val1 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    q_val2 = (val2 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    q_val = q_val1 + q_val2 * 16
    return q_val, scales, zeros


@triton.jit
def _fill_kv_cache_kernel(
    KStates,
    VStates,
    KCaches,
    VCaches,
    QStartLoc,
    QSeqLens,
    KVSeqLens,
    BlockOffsets,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_v: tl.constexpr,
    stride_kss,
    stride_ksh,
    stride_ksd,
    stride_vss,
    stride_vsh,
    stride_vsd,
    stride_kcn: tl.constexpr,
    stride_kcb: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr,
    stride_vcb: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    stride_boff,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):

    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    h_off = tl.arange(0, BLOCK_H)
    d_off = tl.arange(0, BLOCK_D)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    block0_first_tokenloc = history_seqlen % BLOCK

    state_token_offset = tl.maximum(block_id * BLOCK - block0_first_tokenloc, 0)
    kv_block_id = _div_up(history_seqlen + 1, BLOCK) - 1 + block_id
    kv_block_id = min(kv_block_id, stride_boff - 1)
    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    cur_startloc = q_startloc + state_token_offset
    ks_ptr = KStates + cur_startloc * stride_kss
    vs_ptr = VStates + cur_startloc * stride_vss

    kc_ptr = KCaches + block_off * stride_kcn
    vc_ptr = VCaches + block_off * stride_vcn

    c_first_tokenloc = block0_first_tokenloc
    if block_id != 0:
        c_first_tokenloc *= 0
    c_last_tokenloc = tl.minimum(
        BLOCK, q_seqlen + block0_first_tokenloc - block_id * BLOCK
    )

    for bidx in range(c_first_tokenloc, c_last_tokenloc):
        sidx = bidx - c_first_tokenloc
        mask = (h_off[:, None] < num_heads) & (d_off[None, :] < head_dim)
        k = tl.load(
            ks_ptr
            + sidx * stride_kss
            + h_off[:, None] * stride_ksh
            + d_off[None, :] * stride_ksd,
            mask=mask,
        )
        tl.store(
            kc_ptr
            + bidx * stride_kcb
            + h_off[:, None] * stride_kch
            + d_off[None, :] * stride_kcd,
            k,
            mask=mask,
        )

        if BLOCK_DV > 0:
            dv_off = tl.arange(0, BLOCK_DV)
            maskv = (h_off[:, None] < num_heads) & (dv_off[None, :] < head_dim_v)
            v = tl.load(
                vs_ptr
                + sidx * stride_vss
                + h_off[:, None] * stride_vsh
                + dv_off[None, :] * stride_vsd,
                mask=maskv,
            )
            tl.store(
                vc_ptr
                + bidx * stride_vcb
                + h_off[:, None] * stride_vch
                + dv_off[None, :] * stride_vcd,
                v,
                mask=maskv,
            )


@triton.jit
def _fill_kv_cache_quant_kernel(
    KStates,
    VStates,
    KCaches,
    VCaches,
    KScalesZeros,
    VScalesZeros,
    QStartLoc,
    QSeqLens,
    KVSeqLens,
    BlockOffsets,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_v: tl.constexpr,
    stride_kss,
    stride_ksh,
    stride_ksd,
    stride_vss,
    stride_vsh,
    stride_vsd,
    stride_kcn: tl.constexpr,
    stride_kcb: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr,
    stride_vcb: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    stride_kszn: tl.constexpr,
    stride_kszb: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,
    stride_vszn: tl.constexpr,
    stride_vszb: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    quant_policy: tl.constexpr,
    stride_boff,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):

    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    d_off = tl.arange(0, BLOCK_D)

    h_off = tl.arange(0, BLOCK_H)
    szd_off = tl.arange(0, 2)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    block0_first_tokenloc = history_seqlen % BLOCK

    state_token_offset = tl.maximum(block_id * BLOCK - block0_first_tokenloc, 0)
    kv_block_id = _div_up(history_seqlen + 1, BLOCK) - 1 + block_id
    kv_block_id = min(kv_block_id, stride_boff - 1)
    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    cur_startloc = q_startloc + state_token_offset
    ks_ptr = KStates + cur_startloc * stride_kss
    vs_ptr = VStates + cur_startloc * stride_vss

    kc_ptr = KCaches + block_off * stride_kcn
    vc_ptr = VCaches + block_off * stride_vcn

    ksz_ptr = KScalesZeros + block_off * stride_kszn
    vsz_ptr = VScalesZeros + block_off * stride_vszn

    c_first_tokenloc = block0_first_tokenloc
    if block_id != 0:
        c_first_tokenloc *= 0
    c_last_tokenloc = tl.minimum(
        BLOCK, q_seqlen + block0_first_tokenloc - block_id * BLOCK
    )

    for bidx in range(c_first_tokenloc, c_last_tokenloc):
        sidx = bidx - c_first_tokenloc
        mask = (h_off[:, None] < num_heads) & (d_off[None, :] < head_dim)
        if quant_policy == 4:
            k1 = tl.load(
                ks_ptr
                + sidx * stride_kss
                + h_off[:, None] * stride_ksh
                + d_off[None, :] * stride_ksd,
                mask=mask,
            )
            k2 = tl.load(
                ks_ptr
                + sidx * stride_kss
                + h_off[:, None] * stride_ksh
                + d_off[None, :] * stride_ksd
                + head_dim * stride_ksd,
                mask=mask,
            )
            q_k, k_scales, k_zeros = _quant_int4(k1, k2)
        else:
            k = tl.load(
                ks_ptr
                + sidx * stride_kss
                + h_off[:, None] * stride_ksh
                + d_off[None, :] * stride_ksd,
                mask=mask,
            )
            q_k, k_scales, k_zeros = _quant_int8(k)
        tl.store(
            kc_ptr
            + bidx * stride_kcb
            + h_off[:, None] * stride_kch
            + d_off[None, :] * stride_kcd,
            q_k,
            mask=mask,
        )
        tl.store(
            ksz_ptr
            + bidx * stride_kszb
            + h_off[:, None] * stride_kszh
            + szd_off[None, :] * stride_kszd,
            k_scales[:, None],
            mask=(h_off[:, None] < num_heads) & (szd_off[None, :] < 1),
        )
        tl.store(
            ksz_ptr
            + bidx * stride_kszb
            + h_off[:, None] * stride_kszh
            + szd_off[None, :] * stride_kszd,
            k_zeros[:, None],
            mask=(h_off[:, None] < num_heads) & (szd_off[None, :] == 1),
        )

        if BLOCK_DV > 0:
            if quant_policy == 4:
                dv_off = tl.arange(0, BLOCK_DV // 2)
                maskv = (h_off[:, None] < num_heads) & (
                    dv_off[None, :] < head_dim_v // 2
                )
                v1 = tl.load(
                    vs_ptr
                    + sidx * stride_vss
                    + h_off[:, None] * stride_vsh
                    + dv_off[None, :] * stride_vsd,
                    mask=maskv,
                )
                v2 = tl.load(
                    vs_ptr
                    + sidx * stride_vss
                    + h_off[:, None] * stride_vsh
                    + dv_off[None, :] * stride_vsd
                    + head_dim_v // 2 * stride_vsd,
                    mask=maskv,
                )
                q_v, v_scales, v_zeros = _quant_int4(v1, v2)
            else:
                dv_off = tl.arange(0, BLOCK_DV)
                maskv = (h_off[:, None] < num_heads) & (dv_off[None, :] < head_dim_v)
                v = tl.load(
                    vs_ptr
                    + sidx * stride_vss
                    + h_off[:, None] * stride_vsh
                    + dv_off[None, :] * stride_vsd,
                    mask=maskv,
                )
                q_v, v_scales, v_zeros = _quant_int8(v)
            tl.store(
                vc_ptr
                + bidx * stride_vcb
                + h_off[:, None] * stride_vch
                + dv_off[None, :] * stride_vcd,
                q_v,
                mask=maskv,
            )
            tl.store(
                vsz_ptr
                + bidx * stride_vszb
                + h_off[:, None] * stride_vszh
                + szd_off[None, :] * stride_vszd,
                v_scales[:, None],
                mask=(h_off[:, None] < num_heads) & (szd_off[None, :] < 1),
            )
            tl.store(
                vsz_ptr
                + bidx * stride_vszb
                + h_off[:, None] * stride_vszh
                + szd_off[None, :] * stride_vszd,
                v_zeros[:, None],
                mask=(h_off[:, None] < num_heads) & (szd_off[None, :] == 1),
            )


def fill_kv_cache(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    k_caches: torch.Tensor,
    v_caches: torch.Tensor,
    q_start_loc: torch.Tensor,
    q_seq_length: torch.Tensor,
    kv_seq_length: torch.Tensor,
    max_q_seq_length: int,
    block_offsets: torch.Tensor,
    k_scales_zeros: torch.Tensor = None,
    v_scales_zeros: torch.Tensor = None,
    quant_policy: Literal[0, 4, 8] = 0,
):

    block_offsets = block_offsets.contiguous()
    batch_size = block_offsets.size(0)
    block_size, num_heads, head_dim = k_caches.size()[1:]
    head_dim_v = v_states.size(-1)
    max_num_blocks = triton.cdiv(max_q_seq_length, block_size) + 1

    BLOCK = block_size
    BLOCK_H = triton.next_power_of_2(num_heads)
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_DV = triton.next_power_of_2(head_dim_v)
    grid = [batch_size, max_num_blocks]
    kernel_meta = get_kernel_meta(k_states)
    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states,
            v_states,
            k_caches,
            v_caches,
            q_start_loc,
            q_seq_length,
            kv_seq_length,
            block_offsets,
            num_heads=num_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            stride_kss=k_states.stride(-3),
            stride_ksh=k_states.stride(-2),
            stride_ksd=k_states.stride(-1),
            stride_vss=v_states.stride(-3),
            stride_vsh=v_states.stride(-2),
            stride_vsd=v_states.stride(-1),
            stride_kcn=k_caches.stride(0),
            stride_kcb=k_caches.stride(1),
            stride_kch=k_caches.stride(2),
            stride_kcd=k_caches.stride(3),
            stride_vcn=v_caches.stride(0),
            stride_vcb=v_caches.stride(1),
            stride_vch=v_caches.stride(2),
            stride_vcd=v_caches.stride(3),
            stride_boff=block_offsets.stride(0),
            BLOCK=BLOCK,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=BLOCK_DV,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=3,
            **kernel_meta,
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states,
            v_states,
            k_caches,
            v_caches,
            k_scales_zeros,
            v_scales_zeros,
            q_start_loc,
            q_seq_length,
            kv_seq_length,
            block_offsets,
            num_heads=num_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            stride_kss=k_states.stride(-3),
            stride_ksh=k_states.stride(-2),
            stride_ksd=k_states.stride(-1),
            stride_vss=v_states.stride(-3),
            stride_vsh=v_states.stride(-2),
            stride_vsd=v_states.stride(-1),
            stride_kcn=k_caches.stride(0),
            stride_kcb=k_caches.stride(1),
            stride_kch=k_caches.stride(2),
            stride_kcd=k_caches.stride(3),
            stride_vcn=v_caches.stride(0),
            stride_vcb=v_caches.stride(1),
            stride_vch=v_caches.stride(2),
            stride_vcd=v_caches.stride(3),
            stride_kszn=k_scales_zeros.stride(0),
            stride_kszb=k_scales_zeros.stride(1),
            stride_kszh=k_scales_zeros.stride(2),
            stride_kszd=k_scales_zeros.stride(3),
            stride_vszn=v_scales_zeros.stride(0),
            stride_vszb=v_scales_zeros.stride(1),
            stride_vszh=v_scales_zeros.stride(2),
            stride_vszd=v_scales_zeros.stride(3),
            quant_policy=quant_policy,
            stride_boff=block_offsets.stride(0),
            BLOCK=BLOCK,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=BLOCK_DV,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=3,
            **kernel_meta,
        )


def test_fill_kv_cache():

    batch_size = 2
    num_heads = 4
    head_dim = 16
    head_dim_v = 16
    block_size = 8
    max_q_seq_length = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_states = torch.rand(
        (batch_size, max_q_seq_length, num_heads, head_dim), dtype=torch.float32
    ).to(device)
    v_states = torch.rand(
        (batch_size, max_q_seq_length, num_heads, head_dim_v), dtype=torch.float32
    ).to(device)
    k_caches = torch.zeros(
        (batch_size, block_size, num_heads, head_dim), dtype=torch.uint8
    ).to(device)
    v_caches = torch.zeros(
        (batch_size, block_size, num_heads, head_dim_v), dtype=torch.uint8
    ).to(device)
    q_start_loc = torch.zeros(batch_size, dtype=torch.int32).to(device)
    q_seq_length = torch.full((batch_size,), max_q_seq_length, dtype=torch.int32).to(
        device
    )
    kv_seq_length = torch.full((batch_size,), max_q_seq_length, dtype=torch.int32).to(
        device
    )
    block_offsets = torch.zeros(
        (batch_size, max_q_seq_length // block_size + 1), dtype=torch.int32
    ).to(device)
    k_scales_zeros = torch.zeros(
        (batch_size, block_size, num_heads, 2), dtype=torch.float32
    ).to(device)
    v_scales_zeros = torch.zeros(
        (batch_size, block_size, num_heads, 2), dtype=torch.float32
    ).to(device)

    results = {}

    fill_kv_cache(
        k_states,
        v_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        max_q_seq_length,
        block_offsets,
        quant_policy=0,
    )
    results["test_case_1"] = (k_caches.clone(), v_caches.clone())

    fill_kv_cache(
        k_states,
        v_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        max_q_seq_length,
        block_offsets,
        k_scales_zeros,
        v_scales_zeros,
        quant_policy=4,
    )
    results["test_case_2"] = (k_caches.clone(), v_caches.clone())

    fill_kv_cache(
        k_states,
        v_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        max_q_seq_length,
        block_offsets,
        k_scales_zeros,
        v_scales_zeros,
        quant_policy=8,
    )
    results["test_case_3"] = (k_caches.clone(), v_caches.clone())

    return results


result_gold = test_fill_kv_cache()
