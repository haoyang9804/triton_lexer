import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def prefix_prefill_fwd_3d(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    cur_batch,
    cur_head,
    start_m,
    cur_batch_in_all_start_index,
    cur_batch_in_all_stop_index,
    cur_batch_query_len,
    B_Start_Loc,
    B_Seqlen,
    Alibi_slopes,
    block_size,
    x,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: int,
    IN_PRECISION: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):

    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)

    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(Alibi_slopes + cur_head)
        alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
        alibi_start_k = 0

    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        bn = tl.load(
            B_Loc
            + cur_batch * stride_b_loc_b
            + ((start_n + offs_n) // block_size) * stride_b_loc_s,
            mask=(start_n + offs_n) < cur_batch_ctx_len,
            other=0,
        )

        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + ((start_n + offs_n[None, :]) % block_size) * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )

        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        )
        k_load = tl.load(
            K_cache + off_k,
            mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
            other=0.0,
        )

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        qk *= sm_scale

        if USE_ALIBI_SLOPES:
            alibi = (
                tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
            ) * alibi_slope
            alibi = tl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi,
                float("-inf"),
            )
            qk += alibi
            alibi_start_k += BLOCK_N

        if SLIDING_WINDOW > 0:

            qk = tl.where(
                (cur_batch_ctx_len + offs_m[:, None]) - (start_n + offs_n[None, :])
                < SLIDING_WINDOW,
                qk,
                -10000,
            )

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        v_load = tl.load(
            V_cache + off_v,
            mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
            other=0.0,
        )
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)

        l_i = l_i_new
        m_i = m_i_new

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

    if USE_ALIBI_SLOPES:
        alibi_start_k = cur_batch_ctx_len

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_query_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale

        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        if USE_ALIBI_SLOPES:
            alibi = (
                tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
            ) * alibi_slope
            alibi = tl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi,
                float("-inf"),
            )
            qk += alibi
            alibi_start_k += BLOCK_N

        if SLIDING_WINDOW > 0:
            qk = tl.where(
                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,
                qk,
                -10000,
            )

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_query_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)

        l_i = l_i_new
        m_i = m_i_new

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len)
    )
    return


@triton.jit
def kernel_paged_attention_2d(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    scale,
    k_scale,
    v_scale,
    seq_idx,
    query_head_idx,
    cur_batch_in_all_start_index,
    cur_batch_in_all_stop_index,
    cur_batch_query_len,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.constexpr,
    query_stride_0: tl.constexpr,
    query_stride_1: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.constexpr,
    stride_k_cache_1: tl.constexpr,
    stride_k_cache_2: tl.constexpr,
    stride_k_cache_3: tl.constexpr,
    stride_k_cache_4: tl.constexpr,
    stride_v_cache_0: tl.constexpr,
    stride_v_cache_1: tl.constexpr,
    stride_v_cache_2: tl.constexpr,
    stride_v_cache_3: tl.constexpr,
):

    kv_head_idx = query_head_idx // num_queries_per_kv

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0 + query_head_idx * query_stride_1
    )

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED),
        mask=dim_mask,
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([1], float("-inf"), dtype=tl.float32)
    L = tl.full([1], 1.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)

        v_offset = (
            physical_block_idx * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[:, None] * stride_v_cache_2
            + offs_n[None, :] * stride_v_cache_3
        )

        k_offset = (
            physical_block_idx * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + offs_n[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[:, None], other=0.0)

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        tmp = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        mask_new = tmp < boundary

        S = tl.where(mask_new, 0.0, float("-inf")).to(tl.float32)
        S += scale * tl.sum(K * Q[:, None], axis=0)

        if SLIDING_WINDOW > 0:
            S = tl.where((seq_len - 1 - tmp) < SLIDING_WINDOW, S, -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope * (tmp - seq_len + 1)

        m_j = tl.maximum(M, tl.max(S, axis=0))

        P = tl.exp(S - m_j)

        l_j = tl.sum(P, axis=0)

        alpha = tl.exp(M - m_j)

        acc = acc * alpha

        L = L * alpha + l_j
        M = m_j

        acc += tl.sum(V * P[None, :], axis=1)

    acc = acc / L

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset + tl.arange(0, HEAD_SIZE_PADDED), acc, mask=dim_mask
    )


@triton.jit
def fused_chunked_prefill_kernel_25d(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    key_ptr,
    value_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    scale,
    k_scale,
    v_scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride_0: tl.constexpr,
    block_table_stride_1: tl.constexpr,
    query_stride_0: tl.constexpr,
    query_stride_1: tl.constexpr,
    query_stride_2: tl.constexpr,
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    output_stride_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.constexpr,
    stride_k_cache_1: tl.constexpr,
    stride_k_cache_2: tl.constexpr,
    stride_k_cache_3: tl.constexpr,
    stride_k_cache_4: tl.constexpr,
    stride_v_cache_0: tl.constexpr,
    stride_v_cache_1: tl.constexpr,
    stride_v_cache_2: tl.constexpr,
    stride_v_cache_3: tl.constexpr,
    stride_k_0: tl.constexpr,
    stride_k_1: tl.constexpr,
    stride_k_2: tl.constexpr,
    stride_v_0: tl.constexpr,
    stride_v_1: tl.constexpr,
    stride_v_2: tl.constexpr,
    query_start_len_ptr,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    start_m = tl.program_id(2)
    kv_head_idx = query_head_idx // num_queries_per_kv

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if cur_batch_query_len > 1:
        prefix_prefill_fwd_3d(
            query_ptr,
            key_ptr,
            value_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            scale,
            k_scale,
            v_scale,
            seq_idx,
            query_head_idx,
            start_m,
            cur_batch_in_all_start_index,
            cur_batch_in_all_stop_index,
            cur_batch_query_len,
            query_start_len_ptr,
            seq_lens_ptr,
            alibi_slopes_ptr,
            BLOCK_SIZE,
            x,
            output_ptr,
            block_table_stride_0,
            block_table_stride_1,
            query_stride_0,
            query_stride_1,
            query_stride_2,
            stride_k_0,
            stride_k_1,
            stride_k_2,
            stride_v_0,
            stride_v_1,
            stride_v_2,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            stride_k_cache_4,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            num_queries_per_kv=num_queries_per_kv,
            IN_PRECISION=IN_PRECISION,
            BLOCK_DMODEL=HEAD_SIZE,
            BLOCK_DMODEL_PADDED=HEAD_SIZE_PADDED,
            USE_ALIBI_SLOPES=USE_ALIBI_SLOPES,
            SLIDING_WINDOW=SLIDING_WINDOW,
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
        )
    else:

        if start_m > 0:
            return
        kernel_paged_attention_2d(
            output_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            seq_lens_ptr,
            alibi_slopes_ptr,
            scale,
            k_scale,
            v_scale,
            seq_idx,
            query_head_idx,
            cur_batch_in_all_start_index,
            cur_batch_in_all_stop_index,
            cur_batch_query_len,
            num_query_heads,
            num_queries_per_kv,
            block_table_stride_0,
            query_stride_0,
            query_stride_1,
            output_stride_0,
            output_stride_1,
            BLOCK_SIZE,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
            USE_ALIBI_SLOPES,
            SLIDING_WINDOW,
            x,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            stride_k_cache_4,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
        )


BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64
NUM_WARPS = 4 if current_platform.is_rocm() else 8

IS_TURING = current_platform.get_device_capability() == (7, 5)


def fused_chunked_prefill_paged_decode(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
):

    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[1] ** 0.5)

    use_alibi_slopes = alibi_slopes is not None

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if "fp8" in kv_cache_dtype:
        assert key_cache.dtype == torch.uint8
        assert value_cache.dtype == torch.uint8

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)
    if (
        key_cache.dtype == torch.uint8
        or value_cache.dtype == torch.uint8
        and kv_cache_dtype == "auto"
    ):
        raise ValueError(
            "kv_cache_dtype='auto' unsupported for\
            FP8 KV Cache prefill kernel"
        )

    q_dtype_is_f32 = query.dtype is torch.float32

    BLOCK = BASE_BLOCK // 2 if q_dtype_is_f32 else BASE_BLOCK

    IN_PRECISION = "ieee" if IS_TURING and q_dtype_is_f32 else None

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    num_queries_per_kv = query.shape[1] // key.shape[1]
    head_size = query.shape[2]

    grid = (
        num_seqs,
        num_query_heads,
        triton.cdiv(max_query_len, BLOCK),
    )

    fused_chunked_prefill_kernel_25d[grid](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        key_ptr=key,
        value_ptr=value,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=alibi_slopes,
        scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride_0=block_table.stride(0),
        block_table_stride_1=block_table.stride(1),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        query_stride_2=query.stride(2),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        output_stride_2=output.stride(2),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        SLIDING_WINDOW=sliding_window,
        x=key_cache.shape[4] if len(key_cache.shape) == 5 else 1,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4) if len(key_cache.shape) == 5 else 1,
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        stride_k_0=key.stride(0),
        stride_k_1=key.stride(1),
        stride_k_2=key.stride(2),
        stride_v_0=value.stride(0),
        stride_v_1=value.stride(1),
        stride_v_2=value.stride(2),
        query_start_len_ptr=query_start_loc,
        IN_PRECISION=IN_PRECISION,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
    )
