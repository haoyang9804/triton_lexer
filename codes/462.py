import torch, math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd
from typing import List, Optional, Union
import torch.nn.functional as F


@triton.jit
def _flash_decoding_stage2_kernel(
    Mid_O,
    Mid_O_LogExpSum,
    Ouput,
    mido_batch_stride,
    mido_heads_stride,
    mido_partitions_stride,
    mido_dim_stride,
    mido_les_batch_stride,
    mido_les_heads_stride,
    mido_les_partitions_stride,
    o_bs_stride,
    o_heads_stride,
    o_dim_stride,
    actual_seq_len,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_part_v = (
        batch_idx * mido_batch_stride
        + head_idx * mido_heads_stride
        + offs_d * mido_dim_stride
    )

    offs_part_max = batch_idx * mido_les_batch_stride + head_idx * mido_les_heads_stride

    part_v_ptrs = Mid_O + offs_part_v
    part_max_ptrs = Mid_O_LogExpSum + offs_part_max

    d_i = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    num_partitions = (actual_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    for _ in range(0, num_partitions, 1):
        part_v = tl.load(part_v_ptrs)
        part_max = tl.load(part_max_ptrs)

        m_ij = tl.maximum(part_max, m_i)
        p = tl.exp(part_v - m_ij)

        alpha = tl.exp(m_i - m_ij)

        d_i = d_i * alpha + p

        acc *= alpha
        acc += p * part_v

        m_i = m_ij
        part_v_ptrs += mido_partitions_stride
        part_max_ptrs += mido_les_partitions_stride

    offs_out = (
        batch_idx * o_bs_stride + head_idx * o_heads_stride + offs_d * o_dim_stride
    )
    tl.store(Ouput + offs_out, acc / d_i)


@torch.no_grad()
def flash_decode_stage2(
    mid_o,
    mid_o_logexpsum,
    atten_output,
    actual_seq_len,
    PARTITION_SIZE,
):
    HEAD_DIM = mid_o.shape[-1]

    batchs, num_heads = mid_o.shape[0], mid_o.shape[1]
    grid = (batchs, num_heads)

    _flash_decoding_stage2_kernel[grid](
        mid_o,
        mid_o_logexpsum,
        atten_output,
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        *atten_output.stride(),
        actual_seq_len,
        BLOCK_DMODEL=HEAD_DIM,
        BLOCK_SEQ=PARTITION_SIZE,
        num_warps=4,
        num_stages=2,
    )


import torch


def pytorch_flash_decode_stage2(mid_o, mid_o_logexpsum, actual_seq_len, partition_size):
    batchs, num_heads, seq_block_num, head_dim = mid_o.shape
    atten_output_pt = torch.zeros(
        batchs, num_heads, head_dim, device="cuda", dtype=torch.float32
    )

    for batch in range(batchs):
        for head in range(num_heads):
            d_i = torch.zeros(head_dim, device="cuda", dtype=torch.float32)
            m_i = torch.full(
                (head_dim,), -float("inf"), device="cuda", dtype=torch.float32
            )
            acc = torch.zeros(head_dim, device="cuda", dtype=torch.float32)
            for partition in range(seq_block_num):
                part_v = mid_o[batch, head, partition]
                part_max = mid_o_logexpsum[batch, head, partition].item()

                part_max_tensor = torch.full(
                    (head_dim,), part_max, device="cuda", dtype=torch.float32
                )
                m_ij = torch.maximum(part_max_tensor, m_i)
                p = torch.exp(part_v - m_ij)

                alpha = torch.exp(m_i - m_ij)

                d_i = d_i * alpha + p
                acc = acc * alpha + p * part_v

                m_i = m_ij

            mask = d_i > 0
            atten_output_pt[batch, head][mask] = acc[mask] / d_i[mask]
            atten_output_pt[batch, head][~mask] = 0.0

    return atten_output_pt


torch.manual_seed(42)


batchs, num_heads, seq_block_num, head_dim = (
    2,
    4,
    4,
    64,
)
actual_seq_len = 128
partition_size = 32


mid_o = torch.randn(
    batchs, num_heads, seq_block_num, head_dim, device="cuda", dtype=torch.float32
)
mid_o_logexpsum = torch.randn(
    batchs, num_heads, seq_block_num, device="cuda", dtype=torch.float32
)


atten_output = torch.zeros(
    batchs, num_heads, head_dim, device="cuda", dtype=torch.float32
)


flash_decode_stage2(
    mid_o,
    mid_o_logexpsum,
    atten_output,
    actual_seq_len=actual_seq_len,
    PARTITION_SIZE=partition_size,
)


pt_atten_output = pytorch_flash_decode_stage2(
    mid_o, mid_o_logexpsum, actual_seq_len, partition_size
)


diff_atten_output = torch.abs(atten_output - pt_atten_output).max()
print(f"Difference in Atten_Output: {diff_atten_output.item()}")


assert diff_atten_output < 1e-3, "Atten_Output 的差异超出容忍范围"
print("Triton 内核与 PyTorch 实现的数值对比通过。")
