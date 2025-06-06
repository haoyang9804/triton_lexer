import torch, math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd


from triton import Config

attention_configs = [
    Config(
        {"BLOCK_M_SIZE": block_m, "BLOCK_N_SIZE": block_n},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(
    configs=attention_configs, key=["BLOCK_DHEAD_SIZE", "heads", "num_kv_groups"]
)
@triton.jit
def flash_attention2_nopad_kernel(
    Q,
    K,
    V,
    O,
    B_Start_Loc,
    B_Seqlen,
    sm_scale,
    heads,
    num_kv_groups,
    stride_q_bs,
    stride_q_heads,
    stride_q_dim,
    stride_k_bs,
    stride_k_heads,
    stride_k_dim,
    stride_v_bs,
    stride_v_heads,
    stride_v_dim,
    stride_o_bs,
    stride_o_heads,
    stride_o_dim,
    BLOCK_DHEAD_SIZE: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):

    block_m_idx = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch_idx = cur_bh // heads
    cur_head_idx = cur_bh % heads
    cur_kv_head_idx = cur_head_idx // num_kv_groups

    cur_seq_len = tl.load(B_Seqlen + cur_batch_idx)

    cur_seq_start_loc = tl.load(B_Start_Loc + cur_batch_idx)

    block_start_loc = block_m_idx * BLOCK_M_SIZE

    offs_n = tl.arange(0, BLOCK_N_SIZE)
    offs_d = tl.arange(0, BLOCK_DHEAD_SIZE)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M_SIZE)

    q_offs = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_q_bs
        + cur_head_idx * stride_q_heads
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < cur_seq_len, other=0.0)

    k_offs = (
        offs_n[None, :] * stride_k_bs
        + cur_kv_head_idx * stride_k_heads
        + offs_d[:, None] * stride_k_dim
    )
    v_offs = (
        offs_n[:, None] * stride_v_bs
        + cur_kv_head_idx * stride_v_heads
        + offs_d[None, :] * stride_v_dim
    )

    k_ptrs = K + k_offs
    v_ptrs = V + v_offs

    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M_SIZE, cur_seq_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_N_SIZE)

        k = tl.load(
            k_ptrs + (cur_seq_start_loc + start_n) * stride_k_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0.0,
        )

        qk = tl.dot(q, k)

        casual_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(casual_mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        d_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        d_i = d_i * alpha + d_ij

        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptrs + (cur_seq_start_loc + start_n) * stride_v_bs,
            mask=(start_n + offs_n[:, None]) < block_end_loc,
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        m_i = m_ij

    acc = acc / d_i[:, None]
    off_o = (
        (cur_seq_start_loc + offs_m[:, None]) * stride_o_bs
        + cur_head_idx * stride_o_heads
        + offs_d[None, :] * stride_o_dim
    )
    out_ptrs = O + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_seq_len)


@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def flash_attention2_no_pad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale,
    b_start_loc,
    b_seq_len,
    max_seq_len,
):

    BLOCK_SIZE = 64
    output = torch.empty_like(q)
    batchs = b_seq_len.shape[0]
    n_heads, HEAD_DIM = q.shape[1], q.shape[2]

    num_kv_groups = q.shape[1] // k.shape[1]
    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE), batchs * n_heads, 1)
    num_warps = 2 if HEAD_DIM <= 64 else 4
    num_stages = 1
    flash_attention2_nopad_kernel[grid](
        q,
        k,
        v,
        output,
        b_start_loc,
        b_seq_len,
        sm_scale,
        n_heads,
        num_kv_groups,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_DHEAD_SIZE=HEAD_DIM,
    )
    return output


def _naive_attention(q, k, v):
    import math

    bs, seqlen, num_head, head_dim = q.shape
    device = q.device
    mask = 1.0 - torch.tril(
        torch.ones((seqlen, seqlen), device=device), diagonal=0
    ).unsqueeze(0).unsqueeze(0)
    mask.masked_fill_(mask.to(torch.bool), -100000000.0)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    scores = torch.nn.functional.softmax(scores.float() + mask, dim=-1).to(q.dtype)
    output = (
        torch.matmul(scores, v)
        .transpose(1, 2)
        .contiguous()
        .reshape(bs, seqlen, num_head, head_dim)
    )
    return output


def _sdpa(q, k, v):
    bs, seqlen, num_head, head_dim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    output = output.transpose(1, 2).contiguous().reshape(bs, seqlen, num_head, head_dim)
    return output


def standard_attention_prefill(q, k, v, b_start_loc, b_seq_len, sdpa=True):
    out = torch.empty_like(q)
    Z = b_start_loc.shape[0]
    for i in range(Z):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        qi = q[start:end].unsqueeze(0)
        ki = k[start:end].unsqueeze(0)
        vi = v[start:end].unsqueeze(0)
        if sdpa:
            oi = _sdpa(qi, ki, vi)
        else:
            oi = _naive_attention(qi, ki, vi)
        out[start:end] = oi.squeeze(0)
    return out


def run_flash_attention2_no_pad_benchmark(
    batch=4, n_heads=32, head_dim=128, max_seq_len_list=[1024, 2048, 4096]
):

    import matplotlib.pyplot as plt

    device = "cuda"
    sm_scale = 1.0 / math.sqrt(head_dim) * 1.4426950408889634
    max_seq_len = max_seq_len_list[0]

    shape = (batch * max_seq_len, n_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn(shape, device=device, dtype=torch.float16)
    v = torch.randn(shape, device=device, dtype=torch.float16)

    b_seq_len = torch.tensor([512, 1024, 512, 1024], dtype=torch.int32, device="cuda")
    b_start_loc = torch.tensor([0, 512, 1536, 2048], dtype=torch.int32, device="cuda")

    triton_output = flash_attention2_no_pad(
        q, k, v, sm_scale, b_start_loc, b_seq_len, max_seq_len
    )
    torch_output = standard_attention_prefill(
        q, k, v, b_start_loc, b_seq_len, sdpa=False
    )
    print(
        f"The maximum difference between torch and triton is {torch.max(torch.abs(torch_output - triton_output))}"
    )

    flash_times = []
    standard_times = []
    iterations = 50

    for seq_len in max_seq_len_list:

        shape = (batch * seq_len, n_heads, head_dim)
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)

        b_start_loc = torch.tensor(
            [0, seq_len, 2 * seq_len, 3 * seq_len], dtype=torch.int32, device="cuda"
        )
        b_seq_len = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

        _ = flash_attention2_no_pad(q, k, v, sm_scale, b_start_loc, b_seq_len, seq_len)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            _ = flash_attention2_no_pad(
                q, k, v, sm_scale, b_start_loc, b_seq_len, seq_len
            )
        end_event.record()
        torch.cuda.synchronize()
        flash_time = start_event.elapsed_time(end_event) / iterations
        flash_times.append(flash_time)

        _ = standard_attention_prefill(q, k, v, b_start_loc, b_seq_len)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(iterations):
            _ = standard_attention_prefill(q, k, v, b_start_loc, b_seq_len)
        end_event.record()
        torch.cuda.synchronize()
        standard_time = start_event.elapsed_time(end_event) / iterations
        standard_times.append(standard_time)

        print(
            f"max_seq_len = {seq_len:4d}: flash_attn = {flash_time:.3f} ms, standard_attn = {standard_time:.3f} ms"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(max_seq_len_list, flash_times, marker="o", label="Flash Attentionv2")
    plt.plot(max_seq_len_list, standard_times, marker="s", label="Standard Attention")
    plt.xlabel("max_seq_len (kv cache length)")
    plt.ylabel("Average execution time (ms)")
    plt.title("Prefill Stage Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("./flashattentionv2_nopad_benchamrk.png")

    return {
        "max_seq_len_list": max_seq_len_list,
        "flash_times": flash_times,
        "standard_times": standard_times,
    }


if __name__ == "__main__":
    stats = run_flash_attention2_no_pad_benchmark()
    print("Benchmark statistics:", stats)
