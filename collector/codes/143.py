import os
import torch

import triton
import triton.language as tl
import triton_dejavu


class MetaData:
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    dropout_p, return_encoded_softmax = 0.0, False

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k

        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(
                cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q
            )
            self.max_seqlens_k = max(
                cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k
            )

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(dropout_p, return_encoded_softmax):
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()
        if self.varlen:
            assert q.dim() == 3
            total_q, nheads_q, head_size = q.shape
            total_k, nheads_k, _ = k.shape
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) >= 2
            assert self.num_contexts >= 0
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)

            assert self.bias == None

            assert self.dropout_p == 0.0
            assert not self.return_encoded_softmax
        else:
            assert q.dim() == 4
            batch, nheads_q, seqlen_q, head_size = q.shape
            _, nheads_k, seqlen_k, _ = k.shape
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]

        assert q.dtype == k.dtype and q.dtype == v.dtype

        assert head_size <= 128
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(
        philox_seed, philox_offset, dropout_p, m, n, stride
    ).to(tl.uint32)

    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


@triton.jit
def load_fn(block_ptr, first, second, pad):
    if first and second:
        tensor = tl.load(block_ptr, boundary_check=(0, 1), padding_option=pad)
    elif first:
        tensor = tl.load(block_ptr, boundary_check=(0,), padding_option=pad)
    elif second:
        tensor = tl.load(block_ptr, boundary_check=(1,), padding_option=pad)
    else:
        tensor = tl.load(block_ptr)
    return tensor


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    actual_seqlen_k,
    actual_seqlen_q,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    encoded_softmax_block_ptr,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    bias_ptr,
    alibi_slope,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):

    for start_n in range(block_min, block_max, BLOCK_N):

        k = load_fn(
            K_block_ptr, PADDED_HEAD, MASK_STEPS and (n_extra_tokens != 0), "zero"
        )
        if PRE_LOAD_V:
            v = load_fn(
                V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero"
            )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        if MASK_STEPS:

            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        qk += tl.dot(q, k)
        if bias_ptr is not None:
            bias = load_fn(
                bias_ptr, False, MASK_STEPS and (n_extra_tokens != 0), "zero"
            )

            qk += bias * 1.44269504089

        if alibi_slope is not None:

            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)

            relative_pos_block = (
                global_m_positions[:, None]
                + actual_seqlen_k
                - global_n_positions[None, :]
                - actual_seqlen_q
            )
            relative_pos_block = tl.abs(relative_pos_block)

            alibi_block = -1 * alibi_slope * relative_pos_block

            qk += alibi_block * 1.44269504089

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = (
                batch_philox_offset
                + start_m * BLOCK_M * actual_seqlen_k
                + start_n
                - BLOCK_N
            )
            keep = dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k
            )
            if RETURN_ENCODED_SOFTMAX:
                tl.store(
                    encoded_softmax_block_ptr,
                    tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty),
                )
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(
                encoded_softmax_block_ptr,
                p.to(encoded_softmax_block_ptr.type.element_ty),
            )

        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(
                V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero"
            )

        l_i = l_i * alpha + l_ij

        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(
                encoded_softmax_block_ptr, (0, BLOCK_N)
            )
    return acc, l_i, m_i


gpu_name = torch.cuda.get_device_name()


@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {
            "BLOCK_M": [32, 64, 128],
            "BLOCK_N": [32, 64, 128],
            "PRE_LOAD_V": [True, False],
        },
        kwarg_conditions=[
            lambda kwarg: (kwarg["BLOCK_M"] >= kwarg["BLOCK_N"]),
            lambda kwarg: (kwarg["BLOCK_M"] != 64) or ("H100" not in gpu_name),
            lambda kwarg: (kwarg["BLOCK_N"] != 128) or (kwarg["PRE_LOAD_V"] == False),
            lambda kwarg: (kwarg["PRE_LOAD_V"] == True) or ("H100" in gpu_name),
        ],
        num_warps=[2, 4, 8],
        num_stages=[2, 4, 6, 8],
        num_ctas=[1],
        enable_warp_specialization=[False, True],
    ),
    rep=10,
    warmup=5,
    key=[
        "hq",
        "hk",
        "IS_CAUSAL",
        "dropout_p",
        "BLOCK_DMODEL",
        "stride_qz",
        "stride_qh",
        "stride_qm",
        "stride_qk",
        "stride_kz",
        "stride_kh",
        "stride_kn",
        "stride_kk",
        "stride_vz",
        "stride_vh",
        "stride_vn",
        "stride_vk",
        "stride_oz",
        "stride_oh",
        "stride_om",
        "stride_on",
        "stride_bz",
        "stride_bh",
        "stride_bm",
        "stride_bn",
        "stride_az",
        "stride_ah",
        "MAX_SEQLENS_Q",
        "MAX_SEQLENS_K",
        "VARLEN",
        "ACTUAL_BLOCK_DMODEL",
    ],
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    bias,
    sm_scale,
    L,
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
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_az,
    stride_ah,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    hq,
    hk,
    alibi_slopes,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start

        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if IS_CAUSAL:

        n_blocks_seqlen = cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )

        n_blocks = min(n_blocks, n_blocks_seqlen)

        if n_blocks <= 0:
            o_offset = (
                off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
            )
            O_block_ptr = tl.make_block_ptr(
                base=Out + o_offset,
                shape=(seqlen_q, BLOCK_DMODEL),
                strides=(stride_om, stride_on),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_DMODEL),
                order=(1, 0),
            )
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)

            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))
            l_ptrs = L + off_z * hq * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m

            l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
            tl.store(l_ptrs, l)

            return

    is_mqa = hq != hk

    if is_mqa:
        off_h_k = off_h_q % hk
    else:
        off_h_k = off_h_q
    need_padding = False
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        need_padding = True
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        need_padding = True
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_head = ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL

    q_offset = off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_offset = off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(ACTUAL_BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_offset = off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, ACTUAL_BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    if BIAS_TYPE != 0:
        b_offset = off_h_q * stride_bh
        bias_ptr = tl.make_block_ptr(
            base=bias + b_offset,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        bias_ptr = None

    if USE_ALIBI != 0:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0

    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
            base=encoded_softmax + off_h_q * seqlen_q * seqlen_k,
            shape=(seqlen_q, seqlen_k),
            strides=(seqlen_k, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        encoded_softmax_block_ptr = 0

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504089

    q = load_fn(Q_block_ptr, True, padded_head, "zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)

    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:

        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:

        masked_blocks = padded_block_k

    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N

    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            encoded_softmax_block_ptr,
            block_min,
            block_max,
            0,
            0,
            0,
            bias_ptr,
            alibi_slope,
            False,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            False,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            padded_head,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()

    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        K_block_ptr = tl.advance(K_block_ptr, (0, n_full_blocks * BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (n_full_blocks * BLOCK_N, 0))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, n_full_blocks * BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(
                encoded_softmax_block_ptr, (0, n_full_blocks)
            )
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            encoded_softmax_block_ptr,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            bias_ptr,
            alibi_slope,
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            True,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            padded_head,
        )

    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)

    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full(
                (BLOCK_DMODEL,), causal_start_idx, dtype=tl.int32
            )
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    l_ptrs = L + off_z * hq * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m

    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)

        l_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    tl.store(O_block_ptr, acc, boundary_check=(0, 1))


def attn_forward_wrapper(
    q,
    k,
    v,
    causal,
    sm_scale,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    metadata = MetaData(sm_scale=sm_scale)
    if causal:
        metadata.need_causal()

    if len(q.shape) == 3:

        metadata.varlen = True
    if len(k.shape) == 3:

        metadata.varlen = True
    if len(v.shape) == 3:

        metadata.varlen = True
    metadata.max_seqlens_q = max_seqlen_q
    metadata.max_seqlens_k = max_seqlen_k
    metadata.cu_seqlens_q = cu_seqlens_q
    metadata.cu_seqlens_k = cu_seqlens_k
    if cu_seqlens_q is not None:
        metadata.num_contexts = len(cu_seqlens_q) - 1
    o = torch.empty_like(q, dtype=v.dtype)

    metadata.check_args(q, k, v, o)
    if metadata.varlen:
        total_q, nheads_q, head_size = q.shape
        total_k, nheads_k, _ = k.shape
        batch = metadata.num_contexts
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        batch, nheads_q, seqlen_q, head_size = q.shape
        _, nheads_k, seqlen_k, _ = k.shape
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))

    unpadded_head_dims = {32, 64, 128}
    if head_size not in unpadded_head_dims:
        padded_d_model = None
        for i in unpadded_head_dims:
            if i > head_size:
                padded_d_model = i
                break
        assert padded_d_model is not None
    else:
        padded_d_model = head_size

    grid = lambda META: (
        triton.cdiv(metadata.max_seqlens_q, META["BLOCK_M"]),
        nheads_q,
        batch,
    )

    if metadata.return_encoded_softmax:
        encoded_softmax = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], k.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        encoded_softmax = None

    M = torch.empty(
        (batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32
    )

    philox_seed = 0x1BF52
    philox_offset = 0x1D4B42

    if metadata.bias is not None:

        assert metadata.bias.numel() < 2**31
        bias_strides = (
            metadata.bias.stride(0),
            metadata.bias.stride(1),
            metadata.bias.stride(2),
            metadata.bias.stride(3),
        )
    else:
        bias_strides = (0, 0, 0, 0)

    if metadata.alibi_slopes is not None:
        alibi_strides = (
            metadata.alibi_slopes.stride(0),
            metadata.alibi_slopes.stride(1),
        )
    else:
        alibi_strides = (0, 0)

    attn_fwd[grid](
        q,
        k,
        v,
        metadata.bias,
        metadata.sm_scale,
        M,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        dropout_p=metadata.dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        encoded_softmax=encoded_softmax,
        hq=nheads_q,
        hk=nheads_k,
        alibi_slopes=metadata.alibi_slopes,
        ACTUAL_BLOCK_DMODEL=head_size,
        MAX_SEQLENS_Q=metadata.max_seqlens_q,
        MAX_SEQLENS_K=metadata.max_seqlens_k,
        IS_CAUSAL=metadata.causal,
        VARLEN=metadata.varlen,
        BLOCK_DMODEL=padded_d_model,
        BIAS_TYPE=0 if metadata.bias is None else 1,
        USE_ALIBI=0 if metadata.alibi_slopes is None else 1,
        ENABLE_DROPOUT=metadata.dropout_p > 0.0,
        RETURN_ENCODED_SOFTMAX=metadata.return_encoded_softmax,
        BATCH_SIZE=q.shape[0]
    )

    return o
