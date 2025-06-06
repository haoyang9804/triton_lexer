import torch
import triton
import triton.language as tl

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_decoding_stage1_kernel(
    Q,
    K,
    V,
    sm_scale,
    actual_seq_len,
    Mid_O,
    Mid_O_LogExpSum,
    q_bs_stride,
    q_heads_stride,
    q_dim_stride,
    k_bs_stride,
    k_heads_stride,
    k_dim_stride,
    v_bs_stride,
    v_heads_stride,
    v_dim_stride,
    mido_batch_stride,
    mido_heads_stride,
    mido_partitions_stride,
    mido_dim_stride,
    mido_les_batch_stride,
    mido_les_heads_stride,
    mido_les_partitions_stride,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_block_idx = tl.program_id(2)

    cur_batch_start_loc = batch_idx * actual_seq_len

    cur_batch_partition_start_index = seq_block_idx * BLOCK_SEQ
    cur_batch_partition_end_index = tl.minimum(
        actual_seq_len, cur_batch_partition_start_index + BLOCK_SEQ
    )

    num_blocks = (
        cur_batch_partition_end_index - cur_batch_partition_start_index + BLOCK_N - 1
    ) // BLOCK_N

    offs_n = cur_batch_partition_start_index + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_offs = batch_idx * q_bs_stride + head_idx * q_heads_stride + offs_d * q_dim_stride

    k_offs = (
        (cur_batch_start_loc + offs_n[:, None]) * k_bs_stride
        + head_idx * k_heads_stride
        + offs_d[None, :] * k_dim_stride
    )

    v_offs = (
        (cur_batch_start_loc + offs_n[:, None]) * v_bs_stride
        + head_idx * v_heads_stride
        + offs_d[None, :] * v_dim_stride
    )

    q_ptrs = Q + q_offs
    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    q = tl.load(q_ptrs)

    d_i = 0.0
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(num_blocks):
        offs_n_new = start_n * BLOCK_N + offs_n

        k_mask = offs_n_new < cur_batch_partition_end_index

        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

        qk = tl.sum(q * k, axis=1)
        qk = qk * sm_scale
        qk = tl.where(k_mask, qk, float("-inf"))

        current_max = tl.max(qk)
        m_ij = tl.maximum(m_i, current_max)
        qk = qk - m_ij

        p = tl.exp(qk)
        alpha = tl.exp(m_i - m_ij)
        d_i = d_i * alpha + tl.sum(p)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        m_i = m_ij

        k_ptrs += BLOCK_N * k_bs_stride
        v_ptrs += BLOCK_N * v_bs_stride

    need_store = num_blocks > 0

    off_mid_o = (
        batch_idx * mido_batch_stride
        + head_idx * mido_heads_stride
        + seq_block_idx * mido_partitions_stride
        + offs_d * mido_dim_stride
    )

    off_mid_o_les = (
        batch_idx * mido_les_batch_stride
        + head_idx * mido_les_heads_stride
        + seq_block_idx * mido_les_partitions_stride
    )

    part_atten_out = acc / d_i
    logexpsum = m_i + tl.log(d_i)

    part_atten_out = tl.where(need_store, part_atten_out, 0.0)
    logexpsum = tl.where(need_store, logexpsum, float("-inf"))

    tl.store(Mid_O + off_mid_o, part_atten_out, mask=need_store)
    tl.store(Mid_O_LogExpSum + off_mid_o_les, logexpsum, mask=need_store)


@torch.no_grad()
def flash_decode_stage1(
    q,
    k,
    v,
    actual_seq_len,
    mid_o,
    mid_o_logexpsum,
    PARTITION_SIZE,
):
    BLOCK_N_SIZE = 32
    BLOCK_DMODEL = q.shape[-1]
    assert (
        PARTITION_SIZE % BLOCK_N_SIZE == 0
    ), "PARTITION_SIZE 必须是 BLOCK_N_SIZE 的倍数"

    batchs, num_heads, head_dim = q.shape
    sm_scale = 1.0 / (head_dim**0.5)
    grid = (batchs, num_heads, triton.cdiv(actual_seq_len, PARTITION_SIZE))

    _flash_decoding_stage1_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        actual_seq_len,
        mid_o,
        mid_o_logexpsum,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        BLOCK_SEQ=PARTITION_SIZE,
        BLOCK_N=BLOCK_N_SIZE,
        BLOCK_DMODEL=head_dim,
        num_warps=1,
        num_stages=2,
    )


import torch


torch.manual_seed(42)


batchs, num_heads, head_dim, seq_len = 2, 4, 64, 128
partition_size = 32


q = torch.randn(batchs, num_heads, head_dim, device="cuda", dtype=torch.float32)
k = torch.randn(
    batchs * seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32
)
v = torch.randn(
    batchs * seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32
)


mid_o = torch.zeros(
    batchs,
    num_heads,
    (seq_len + partition_size - 1) // partition_size,
    head_dim,
    device="cuda",
    dtype=torch.float32,
)
mid_o_logexpsum = torch.zeros(
    batchs,
    num_heads,
    (seq_len + partition_size - 1) // partition_size,
    device="cuda",
    dtype=torch.float32,
)


flash_decode_stage1(
    q,
    k,
    v,
    actual_seq_len=seq_len,
    mid_o=mid_o,
    mid_o_logexpsum=mid_o_logexpsum,
    PARTITION_SIZE=partition_size,
)


print("Mid_O:", mid_o)
print("Mid_O_LogExpSum:", mid_o_logexpsum)
