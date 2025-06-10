import triton, torch
import triton.language as tl
from torch.cuda.amp import custom_fwd


@triton.jit
def _flash_decoding_stage1_kernel(
    Q,
    K,
    V,
    qk_scale,
    b_req_tokens_table,
    B_Seqlen,
    num_kv_groups,
    Mid_O,
    Mid_O_LogExpSum,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
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

    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    seq_block_pid = tl.program_id(2)
    kv_head_pid = head_pid // num_kv_groups

    cur_batch_seq_len = tl.load(B_Seqlen + batch_pid)
    cur_req_start_loc = tl.load(b_req_tokens_table + stride_req_to_tokens_b * batch_pid)

    cur_batch_partition_start_index = seq_block_pid * BLOCK_SEQ
    cur_batch_partition_end_index = tl.minimum(
        cur_batch_seq_len, cur_batch_partition_start_index + BLOCK_SEQ
    )

    num_blocks = tl.where(
        cur_batch_partition_end_index - cur_batch_partition_start_index <= 0,
        0,
        (cur_batch_partition_end_index - cur_batch_partition_start_index + BLOCK_N - 1)
        // BLOCK_N,
    )

    offs_n = cur_batch_partition_start_index + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_offs = batch_pid * q_bs_stride + head_pid * q_heads_stride + offs_d * q_dim_stride
    k_offs = kv_head_pid * k_heads_stride + offs_d[None, :] * k_dim_stride

    q_ptrs = Q + q_offs
    q = tl.load(q_ptrs)

    d_i = 0.0
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, num_blocks, 1):

        offs_n_new = offs_n + start_n * BLOCK_N
        k_loc = tl.load(
            b_req_tokens_table + stride_req_to_tokens_b * batch_pid + offs_n_new,
            mask=offs_n_new < cur_batch_partition_end_index,
            other=0.0,
        )
        k_ptrs = k_loc[:, None] * k_bs_stride + k_offs

        k_mask = offs_n_new < cur_batch_partition_end_index

        k = tl.load(K + k_ptrs, mask=k_mask[:, None], other=0.0)
        v = tl.load(V + k_ptrs, mask=k_mask[:, None], other=0.0)

        qk = tl.sum(q[None, :] * k, axis=1)
        qk *= qk_scale
        qk = tl.where(k_mask, qk, float("-inf"))

        current_max = tl.max(qk)
        m_ij = tl.maximum(m_i, current_max)
        p = tl.exp(qk - m_ij)

        alpha = tl.exp(m_i - m_ij)
        d_i = alpha * d_i + tl.sum(p, axis=0)

        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)

        m_i = m_ij

    need_store = num_blocks > 0

    off_mid_o = (
        batch_pid * mido_batch_stride
        + head_pid * mido_heads_stride
        + seq_block_pid * mido_partitions_stride
        + offs_d * mido_dim_stride
    )

    off_mid_o_les = (
        batch_pid * mido_les_batch_stride
        + head_pid * mido_les_heads_stride
        + seq_block_pid * mido_les_partitions_stride
    )

    need_store = tl.where(num_blocks == 0, 0, 1)
    for _ in range(0, need_store, 1):
        tl.store(Mid_O + off_mid_o, acc / d_i)
        tl.store(Mid_O_LogExpSum + off_mid_o_les, m_i + tl.log(d_i))


@torch.no_grad()
def flash_decode_stage1(
    q,
    k,
    v,
    qk_scale,
    b_req_tokens_table,
    b_seq_len,
    max_actual_seq_len,
    mid_o,
    mid_o_logexpsum,
    PARTITION_SIZE,
):
    BLOCK_N_SIZE = 16

    assert (
        PARTITION_SIZE % BLOCK_N_SIZE == 0
    ), "PARTITION_SIZE 必须是 BLOCK_N_SIZE 的倍数"

    batchs, num_heads, head_dim = q.shape

    grid = (
        batchs,
        num_heads,
        triton.cdiv(max_actual_seq_len + PARTITION_SIZE - 1, PARTITION_SIZE),
    )
    num_kv_groups = q.shape[1] // k.shape[1]

    _flash_decoding_stage1_kernel[grid](
        q,
        k,
        v,
        qk_scale,
        b_req_tokens_table,
        b_seq_len,
        num_kv_groups,
        mid_o,
        mid_o_logexpsum,
        *b_req_tokens_table.stride(),
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
    B_Seqlen,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):

    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    cur_batch_seq_len = tl.load(B_Seqlen + batch_pid)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_part_v = batch_pid * mido_batch_stride + head_pid * mido_heads_stride + offs_d

    offs_part_max = batch_pid * mido_les_batch_stride + head_pid * mido_les_heads_stride

    part_v_ptrs = Mid_O + offs_part_v
    part_max_ptrs = Mid_O_LogExpSum + offs_part_max

    d_i = 0.0
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    num_partitions = (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    for block_seq_n in range(0, num_partitions, 1):
        part_v = tl.load(part_v_ptrs + block_seq_n * mido_partitions_stride)
        part_max = tl.load(part_max_ptrs + block_seq_n)

        m_ij = tl.maximum(part_max, m_i)

        alpha = tl.exp(m_i - m_ij)

        p = tl.exp(part_max - m_ij)
        acc = alpha * acc + p * part_v

        d_i = alpha * d_i + p

        m_i = m_ij

    offs_out = (
        batch_pid * o_bs_stride + head_pid * o_heads_stride + offs_d * o_dim_stride
    )
    tl.store(Ouput + offs_out, acc / d_i)


@torch.no_grad()
def flash_decode_stage2(
    mid_o,
    mid_o_logexpsum,
    atten_output,
    b_seq_len,
    PARTITION_SIZE,
):
    batchs, num_heads, HEAD_DIM = mid_o.shape[0], mid_o.shape[1], mid_o.shape[-1]
    grid = (batchs, num_heads)

    _flash_decoding_stage2_kernel[grid](
        mid_o,
        mid_o_logexpsum,
        atten_output,
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        *atten_output.stride(),
        b_seq_len,
        BLOCK_DMODEL=HEAD_DIM,
        BLOCK_SEQ=PARTITION_SIZE,
        num_warps=4,
        num_stages=2,
    )


@torch.no_grad()
def flash_decoding(
    q,
    k_cache,
    v_cache,
    qk_scale,
    b_req_tokens_table,
    b_seq_len,
    max_actual_seq_len,
):

    assert q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1]
    PARTITION_SIZE = 128
    batchs, num_heads, head_dim = q.shape

    max_num_partitions = (max_actual_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    mid_o = torch.empty(
        (batchs, num_heads, max_num_partitions, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    mid_o_logexpsum = torch.empty(
        (batchs, num_heads, max_num_partitions), dtype=torch.float32, device=q.device
    )

    flash_decode_stage1(
        q,
        k_cache,
        v_cache,
        qk_scale,
        b_req_tokens_table,
        b_seq_len,
        max_actual_seq_len,
        mid_o,
        mid_o_logexpsum,
        PARTITION_SIZE,
    )

    atten_output = torch.empty_like(q)

    flash_decode_stage2(mid_o, mid_o_logexpsum, atten_output, b_seq_len, PARTITION_SIZE)

    return atten_output


def _naive_attention(q, k, v):
    import math

    head_dim = q.shape[-1]
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    scores = torch.nn.functional.softmax(scores.float(), dim=-1).to(q.dtype)
    output = torch.matmul(scores, v).transpose(0, 1).contiguous()
    return output


def torch_attention_with_kvcache(q, k_cache, v_cache, b_start_loc, b_seq_len):
    out = torch.empty_like(q)
    Z = q.shape[0]
    for i in range(Z):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        q_i = q[i : i + 1]
        k_i = k_cache[start:end]
        v_i = v_cache[start:end]
        o_i = _naive_attention(q_i, k_i, v_i)
        out[i : i + 1] = o_i
    return out


def plot_performance_comparison(token_sizes, warmup_iterations=10, test_iterations=50):

    import matplotlib.pyplot as plt

    device = torch.device("cuda")
    batch = 4
    num_heads = 32
    head_dim = 64
    qk_scale = 1.0 / (head_dim**0.5)
    q = torch.randn(batch * 1, num_heads, head_dim, device=device)

    flash_times = []
    standard_times = []

    for tokens in token_sizes:
        print(f"\n测试 token size: {tokens}")
        k_cache = torch.randn(batch * tokens, num_heads, head_dim, device=device)
        v_cache = torch.randn(batch * tokens, num_heads, head_dim, device=device)
        b_req_tokens_table = torch.arange(
            0, tokens, device=device, dtype=torch.int32
        ).repeat(batch, 1)
        b_start_loc = torch.tensor(
            [0, tokens, 2 * tokens, 3 * tokens], dtype=torch.int32, device="cuda"
        )
        b_seq_len = torch.full((batch,), tokens, device=device, dtype=torch.int32)
        max_actual_seq_len = tokens

        for _ in range(warmup_iterations):
            _ = flash_decoding(
                q,
                k_cache,
                v_cache,
                qk_scale,
                b_req_tokens_table,
                b_seq_len,
                max_actual_seq_len,
            )

        torch.cuda.synchronize()
        flash_start = torch.cuda.Event(enable_timing=True)
        flash_end = torch.cuda.Event(enable_timing=True)
        flash_start.record()
        for _ in range(test_iterations):
            _ = flash_decoding(
                q,
                k_cache,
                v_cache,
                qk_scale,
                b_req_tokens_table,
                b_seq_len,
                max_actual_seq_len,
            )
        flash_end.record()
        torch.cuda.synchronize()
        flash_avg = flash_start.elapsed_time(flash_end) / test_iterations
        flash_times.append(flash_avg)
        print(f"Flash Decoding 平均时间: {flash_avg:.3f} ms")

        for _ in range(warmup_iterations):
            _ = torch_attention_with_kvcache(
                q, k_cache, v_cache, b_start_loc, b_seq_len
            )

        torch.cuda.synchronize()
        std_start = torch.cuda.Event(enable_timing=True)
        std_end = torch.cuda.Event(enable_timing=True)
        std_start.record()
        for _ in range(test_iterations):
            _ = torch_attention_with_kvcache(
                q, k_cache, v_cache, b_start_loc, b_seq_len
            )
        std_end.record()
        torch.cuda.synchronize()
        std_avg = std_start.elapsed_time(std_end) / test_iterations
        standard_times.append(std_avg)
        print(f"Standard Attention 平均时间: {std_avg:.3f} ms")

    plt.figure(figsize=(8, 6))
    plt.plot(token_sizes, flash_times, marker="o", label="Flash Decoding")
    plt.plot(token_sizes, standard_times, marker="o", label="Standard Attention")
    plt.xlabel("Token Size (kv cache length)")
    plt.ylabel("Average Time (ms)")
    plt.title("Performance Comparison: Flash Decoding vs Standard Attention")
    plt.legend()
    plt.grid(True)
    plt.savefig("./flashdecoding_benchamrk.png")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    batch = 4
    num_heads = 32
    head_dim = 64
    max_tokens = 2048
    qk_scale = 1.0 / (head_dim**0.5)

    q = torch.randn(batch * 1, num_heads, head_dim, device=device)
    k_cache = torch.randn(batch * max_tokens, num_heads, head_dim, device=device)
    v_cache = torch.randn(batch * max_tokens, num_heads, head_dim, device=device)

    b_req_tokens_table = torch.arange(
        0, max_tokens * batch, device=device, dtype=torch.int32
    ).view(batch, max_tokens)
    b_seq_len = torch.full((batch,), max_tokens, device=device, dtype=torch.int32)
    b_start_loc = torch.tensor(
        [0, max_tokens, 2 * max_tokens, 3 * max_tokens],
        dtype=torch.int32,
        device="cuda",
    )

    flash_out = flash_decoding(
        q, k_cache, v_cache, qk_scale, b_req_tokens_table, b_seq_len, max_tokens
    )
    standard_out = torch_attention_with_kvcache(
        q, k_cache, v_cache, b_start_loc, b_seq_len
    )
    print("Flash Decoding output shape:", flash_out.shape)
    print("Standard Attention output shape:", standard_out.shape)
    if torch.allclose(flash_out, standard_out, atol=1e-3, rtol=1e-3):
        print("验证通过: Flash Decoding 输出与标准 Attention 接近。")
    else:
        diff = (flash_out - standard_out).abs().max().item()
        print(f"验证失败：最大误差为 {diff:.4f}")

    token_numbers = [64, 128, 256, 512, 1024, max_tokens]
    plot_performance_comparison(token_numbers, warmup_iterations=10, test_iterations=50)


if __name__ == "__main__":
    main()
