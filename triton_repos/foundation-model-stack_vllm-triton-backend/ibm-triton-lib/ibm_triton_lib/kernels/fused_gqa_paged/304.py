import math
import triton.language as tl
import triton
import torch

from .sb_jit_func import get_split_tblocks_range, attend_one_block
from .utils import is_fp8_dtype, compute_split_l


@triton.jit
def triton_fused_gqa_paged_splitkv(
    q_ptr,
    k_ptr,
    v_ptr,
    o_split_ptr,
    m_i_split_ptr,
    l_i_split_ptr,
    pt_ptr,
    context_lens_ptr,
    alibi_slopes_ptr,
    stride_qp,
    stride_qs,
    stride_ki,
    stride_kg,
    stride_kt,
    stride_vi,
    stride_vg,
    stride_vt,
    stride_osp,
    stride_ossl,
    stride_oss,
    stride_mip,
    stride_misl,
    stride_lip,
    stride_lisl,
    stride_ptb,
    stride_asg,
    sm_scale: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_SS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FORCE_FP16_PV: tl.constexpr,
    QUANTIZE_P: tl.constexpr,
    MAX_FP8: tl.constexpr,
    IS_STICKBREAKING: tl.constexpr = False,
    USE_DOT_CUMSUM: tl.constexpr = False,
    TRANSPOSED: tl.constexpr = False,
    USE_ALIBI_SLOPES: tl.constexpr = False,
):

    pid_s = tl.program_id(0)
    pid_p = tl.program_id(1)
    b = pid_p // G
    g = pid_p % G
    pid_sl = tl.program_id(2)
    num_splits = tl.num_programs(2)

    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE_SIZE)
    rng_ss = tl.arange(0, BLOCK_SS)
    qk_scale = sm_scale * 1.44269504

    kv_len = tl.load(context_lens_ptr + b)
    pg_start, pg_end = get_split_tblocks_range(pid_sl, kv_len, PAGE_SIZE, num_splits)
    num_pages = pg_end - pg_start

    for ss in range(tl.cdiv(BLOCK_S, BLOCK_SS)):

        m_i = tl.zeros([BLOCK_SS], dtype=tl.float32) - float("inf")
        d_i = tl.zeros([BLOCK_SS], dtype=tl.float32)
        acc = tl.zeros(
            [BLOCK_SS, D] if not TRANSPOSED else [D, BLOCK_SS], dtype=tl.float32
        )

        offs_ss = pid_s * BLOCK_S + ss * BLOCK_SS + rng_ss
        ss_mask = offs_ss < S

        q_ptrs = (
            q_ptr + pid_p * stride_qp + offs_ss[:, None] * stride_qs + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=ss_mask[:, None], other=0.0)

        alibi_slopes = None
        if USE_ALIBI_SLOPES:
            alibi_slopes = tl.load(
                alibi_slopes_ptr + g * stride_asg + rng_ss, mask=(rng_ss < S)
            )

        for i in range(num_pages):

            pg_offset = i if not IS_STICKBREAKING else ((num_pages - 1) - i)
            pg_idx_ptr = pt_ptr + b * stride_ptb + (pg_start + pg_offset)
            cache_idx = tl.load(pg_idx_ptr)

            k_ptrs = (
                k_ptr
                + cache_idx * stride_ki
                + g * stride_kg
                + offs_t[:, None] * stride_kt
                + offs_d[None, :]
            )
            v_ptrs = (
                v_ptr
                + cache_idx * stride_vi
                + g * stride_vg
                + offs_t[:, None] * stride_vt
                + offs_d[None, :]
            )

            tb_len_max = kv_len % PAGE_SIZE
            if tb_len_max == 0:
                tb_len_max = tb_len_max + PAGE_SIZE
            is_last_block = (pid_sl == (num_splits - 1)) and (
                pg_offset == (num_pages - 1)
            )
            t_mask = (offs_t < tb_len_max) if is_last_block else (offs_t < PAGE_SIZE)

            k = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=t_mask[:, None], other=0.0)

            alibi_distances = None
            if USE_ALIBI_SLOPES:
                alibi_distances = (pg_offset * PAGE_SIZE) + (offs_t + 1) - kv_len

            m_i, d_i, acc = attend_one_block(
                q,
                k,
                v,
                qk_scale,
                m_i,
                d_i,
                acc,
                alibi_slopes,
                alibi_distances,
                IS_LAST_BLOCK=is_last_block,
                tb_len_max=tb_len_max,
                offs_t=offs_t,
                FORCE_FP16_PV=FORCE_FP16_PV,
                QUANTIZE_P=QUANTIZE_P,
                MAX_FP8=MAX_FP8,
                IS_STICKBREAKING=IS_STICKBREAKING,
                USE_DOT_CUMSUM=USE_DOT_CUMSUM,
                TRANSPOSED=TRANSPOSED,
                USE_ALIBI_SLOPES=USE_ALIBI_SLOPES,
                ATTEND_CURRENT=False,
            )

        o_ptrs = (
            o_split_ptr
            + pid_p * stride_osp
            + pid_sl * stride_ossl
            + offs_ss[:, None] * stride_oss
            + offs_d[None, :]
        )
        tl.store(
            o_ptrs,
            (
                acc.to(o_split_ptr.dtype.element_ty)
                if not TRANSPOSED
                else acc.T.to(o_split_ptr.dtype.element_ty)
            ),
            mask=ss_mask[:, None],
        )

        if not IS_STICKBREAKING:
            m_i_ptrs = (
                m_i_split_ptr + pid_p * stride_mip + pid_sl * stride_misl + offs_ss
            )
            tl.store(m_i_ptrs, m_i, mask=ss_mask)

        l_i_ptrs = l_i_split_ptr + pid_p * stride_lip + pid_sl * stride_lisl + offs_ss
        tl.store(l_i_ptrs, d_i, mask=ss_mask)

    return


def torch_fused_gqa_reduce_splitkv(o_split, m_i_split, l_i_split, o_dtype):
    g_m = (m_i_split.max(dim=1, keepdim=True)).values
    alpha = torch.exp2(m_i_split - g_m)
    l_sum = l_i_split * alpha
    g_sum = l_sum.sum(dim=1)
    o = torch.mul(o_split, alpha[:, :, :, None]).sum(dim=1)
    o /= g_sum[:, :, None]
    return o.to(o_dtype)


@triton.jit
def triton_fused_gqa_reduce_splitkv(
    o_ptr,
    o_s_ptr,
    m_i_ptr,
    l_i_ptr,
    stride_op,
    stride_os,
    stride_osp,
    stride_ossl,
    stride_oss,
    stride_mip,
    stride_misl,
    stride_lip,
    stride_lisl,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_SS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    O_DTYPE: tl.constexpr = None,
):

    pid_s = tl.program_id(0)
    pid_p = tl.program_id(1)

    o_dtype = o_s_ptr.dtype.element_ty if O_DTYPE is None else O_DTYPE
    offs_sl = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, D)

    for ss in range(tl.cdiv(BLOCK_S, BLOCK_SS)):
        offs_ss = pid_s * BLOCK_S + ss * BLOCK_SS + tl.arange(0, BLOCK_SS)
        ss_mask = offs_ss < S

        o_s_ptrs = (
            o_s_ptr
            + pid_p * stride_osp
            + offs_sl[:, None, None] * stride_ossl
            + offs_ss[None, :, None] * stride_oss
            + offs_d[None, None, :]
        )
        m_i_ptrs = (
            m_i_ptr
            + pid_p * stride_mip
            + offs_sl[:, None] * stride_misl
            + offs_ss[None, :]
        )
        l_i_ptrs = (
            l_i_ptr
            + pid_p * stride_lip
            + offs_sl[:, None] * stride_lisl
            + offs_ss[None, :]
        )

        o_s = tl.load(o_s_ptrs, mask=ss_mask[None, :, None], other=0.0)
        m_i = tl.load(m_i_ptrs, mask=ss_mask[None, :], other=0.0)
        l_i = tl.load(l_i_ptrs, mask=ss_mask[None, :], other=0.0)

        g_m = tl.max(m_i, axis=0)
        alpha = tl.exp2(m_i - g_m[None, :])
        l_sum = l_i * alpha
        g_sum = tl.sum(l_sum, axis=0)
        o = tl.sum((o_s * alpha[:, :, None]), axis=0)
        o = o / g_sum[:, None]
        o_ptrs = (
            o_ptr + pid_p * stride_op + offs_ss[:, None] * stride_os + offs_d[None, :]
        )
        tl.store(o_ptrs, o.to(o_dtype), mask=ss_mask[:, None])

    return


def torch_fused_gqa_merge_sb_splitkv(o_split, neg_loc_acc_split, o_dtype):
    neg_loc_acc_clone = torch.empty_like(neg_loc_acc_split)
    neg_loc_acc_clone[:, :-1, ...] = neg_loc_acc_split[:, 1:, ...]
    neg_loc_acc_clone[:, -1, ...] = 0
    neg_loc_acc_clone = (
        torch.sum(neg_loc_acc_clone, dim=1, keepdim=True)
        - torch.cumsum(neg_loc_acc_clone, dim=1)
        + neg_loc_acc_clone
    )

    rem_split = torch.exp2(neg_loc_acc_clone)
    o_split = o_split * rem_split[..., None]
    o = o_split.sum(dim=1)
    neg_loc_acc = neg_loc_acc_split.sum(dim=1)
    rem = torch.exp2(neg_loc_acc)

    return o.to(o_dtype), rem, neg_loc_acc


@triton.jit
def triton_fused_gqa_merge_sb_splitkv(
    o_ptr,
    o_s_ptr,
    l_i_ptr,
    v_ptr,
    pt_ptr,
    context_lens_ptr,
    stride_op,
    stride_os,
    stride_osp,
    stride_ossl,
    stride_oss,
    stride_lip,
    stride_lisl,
    stride_vi,
    stride_vg,
    stride_vt,
    stride_ptb,
    G: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_SS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    O_DTYPE: tl.constexpr = None,
    ADD_REM: tl.constexpr = False,
):
    pid_s = tl.program_id(0)
    pid_p = tl.program_id(1)
    b = pid_p // G
    g = pid_p % G

    o_dtype = o_s_ptr.dtype.element_ty if O_DTYPE is None else O_DTYPE
    offs_sl = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, D)

    if ADD_REM:

        v_len = tl.load(context_lens_ptr + b)
        num_pages = (v_len + PAGE_SIZE - 1) // PAGE_SIZE
        cache_idx = tl.load(pt_ptr + b * stride_ptb + (num_pages - 1))

        t_idx = v_len % PAGE_SIZE - 1
        t_idx = t_idx if t_idx > 0 else t_idx + PAGE_SIZE

        v_ptrs = (
            v_ptr + cache_idx * stride_vi + g * stride_vg + t_idx * stride_vt + offs_d
        )
        v = tl.load(v_ptrs)

    for ss in range(tl.cdiv(BLOCK_S, BLOCK_SS)):
        offs_ss = pid_s * BLOCK_S + ss * BLOCK_SS + tl.arange(0, BLOCK_SS)
        ss_mask = offs_ss < S

        o_s_ptrs = (
            o_s_ptr
            + pid_p * stride_osp
            + offs_sl[:, None, None] * stride_ossl
            + offs_ss[None, :, None] * stride_oss
            + offs_d[None, None, :]
        )
        l_i_ptrs = (
            l_i_ptr
            + pid_p * stride_lip
            + offs_sl[:, None] * stride_lisl
            + offs_ss[None, :]
        )

        o_s = tl.load(o_s_ptrs, mask=ss_mask[None, :, None], other=0.0)
        l_i = tl.load(l_i_ptrs, mask=ss_mask[None, :], other=0.0)

        neg_loc_acc_split = tl.cumsum(l_i, axis=0, reverse=True) - l_i
        rem_split = tl.exp2(neg_loc_acc_split)
        o_split = o_s * rem_split[:, :, None]
        o = tl.sum(o_split, axis=0)

        if ADD_REM:
            rem = tl.exp2(tl.sum(l_i, axis=0))
            o += rem[:, None] * v[None, :]

        o_ptrs = (
            o_ptr + pid_p * stride_op + offs_ss[:, None] * stride_os + offs_d[None, :]
        )
        tl.store(o_ptrs, o.to(o_dtype), mask=ss_mask[:, None])

    return


def paged_attention_triton_3d(
    o,
    q,
    k,
    v,
    sm_scale,
    block_tables,
    context_lens,
    alibi_slopes,
    block_size,
    num_seqs,
    num_query_heads,
    num_queries_per_kv,
    head_size,
    o_dtype=None,
    transposed: bool = False,
    force_fp16_pv: bool = False,
    force_split_l: int = None,
    quantize_p: bool = False,
    stickbreaking: bool = False,
    sb_add_rem: bool = False,
    use_torch_2nd_stage_impl: bool = False,
):
    B = num_seqs
    H = num_queries_per_kv
    assert num_query_heads % num_queries_per_kv == 0
    G = num_query_heads // num_queries_per_kv
    Q = 1
    D = head_size
    PAGE_SIZE = block_size

    L = block_tables.shape[1] * PAGE_SIZE
    o_dtype = q.dtype if o_dtype == None else o_dtype

    P = B * G

    S = H * Q
    q = q.reshape([P, S, D])
    device = q.device

    USE_ALIBI_SLOPES = False
    if alibi_slopes is None:
        alibi_slopes = torch.empty([1, 1], dtype=torch.float32, device=device)
    else:
        USE_ALIBI_SLOPES = True
        alibi_slopes = alibi_slopes.reshape([G, H])

    BLOCK_SS = 64 if (is_fp8_dtype(k.dtype) and not transposed) else 16

    NUM_SS = 1
    BLOCK_S = min(BLOCK_SS * NUM_SS, S)

    NUM_SPLITS = (
        compute_split_l(L, PAGE_SIZE, P) if force_split_l == None else force_split_l
    )

    o_split = torch.empty([P, NUM_SPLITS, S, D], dtype=o_dtype, device=device)
    m_i_split = torch.empty([P, NUM_SPLITS, S], dtype=torch.float32, device=device)
    l_i_split = torch.empty([P, NUM_SPLITS, S], dtype=torch.float32, device=device)

    grid = (math.ceil(S / BLOCK_S), P, NUM_SPLITS)
    assert not stickbreaking or Q == 1

    _kernel = triton_fused_gqa_paged_splitkv
    _kernel[grid](
        q,
        k,
        v,
        o_split,
        m_i_split,
        l_i_split,
        block_tables,
        context_lens,
        alibi_slopes,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o_split.stride(0),
        o_split.stride(1),
        o_split.stride(2),
        m_i_split.stride(0),
        m_i_split.stride(1),
        l_i_split.stride(0),
        l_i_split.stride(1),
        block_tables.stride(0),
        alibi_slopes.stride(0),
        sm_scale,
        G,
        D,
        S,
        BLOCK_S,
        BLOCK_SS,
        PAGE_SIZE,
        FORCE_FP16_PV=force_fp16_pv,
        QUANTIZE_P=quantize_p,
        MAX_FP8=torch.finfo(torch.float8_e4m3fn).max,
        IS_STICKBREAKING=stickbreaking,
        TRANSPOSED=transposed,
        num_stages=(2 if k.dtype != torch.float8_e4m3fn else 3),
        USE_ALIBI_SLOPES=USE_ALIBI_SLOPES,
    )

    if use_torch_2nd_stage_impl:
        if not stickbreaking:
            output = torch_fused_gqa_reduce_splitkv(
                o_split=o_split,
                m_i_split=m_i_split,
                l_i_split=l_i_split,
                o_dtype=o_dtype,
            )
        else:
            assert (
                not sb_add_rem
            ), "add remainder not supported by the torch 2nd stage impl"
            output, rem, _ = torch_fused_gqa_merge_sb_splitkv(
                o_split=o_split,
                neg_loc_acc_split=l_i_split,
                o_dtype=o_dtype,
            )

        output.copy_(o.reshape(output.shape))
    else:
        grid2 = (math.ceil(S / BLOCK_S), P, 1)
        o = o.reshape([P, S, D])
        if not stickbreaking:
            triton_fused_gqa_reduce_splitkv[grid2](
                o,
                o_split,
                m_i_split,
                l_i_split,
                o.stride(0),
                o.stride(1),
                o_split.stride(0),
                o_split.stride(1),
                o_split.stride(2),
                m_i_split.stride(0),
                m_i_split.stride(1),
                l_i_split.stride(0),
                l_i_split.stride(1),
                D,
                S,
                BLOCK_S,
                BLOCK_SS,
                NUM_SPLITS,
            )
        else:
            triton_fused_gqa_merge_sb_splitkv[grid2](
                o,
                o_split,
                l_i_split,
                v,
                block_tables,
                context_lens,
                o.stride(0),
                o.stride(1),
                o_split.stride(0),
                o_split.stride(1),
                o_split.stride(2),
                l_i_split.stride(0),
                l_i_split.stride(1),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                block_tables.stride(0),
                G,
                D,
                S,
                BLOCK_S,
                BLOCK_SS,
                PAGE_SIZE,
                NUM_SPLITS,
                ADD_REM=sb_add_rem,
            )

    return
