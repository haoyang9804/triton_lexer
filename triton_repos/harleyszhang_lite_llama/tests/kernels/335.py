import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)

    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    cos_row_idx = pid % (sl)
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)

    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
        sin_row.dtype
    )

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def rope_forward(q, k, cos, sin):

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_rope[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q, k


def torch_rotary_emb(x, cos, sin):
    seq_len, h, d = x.shape

    half_dim = cos.shape[-1]
    x0 = x[:, :, :half_dim]
    x1 = x[:, :, half_dim : 2 * half_dim]

    cos = cos.view(seq_len, 1, half_dim)
    sin = sin.view(seq_len, 1, half_dim)

    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos

    if 2 * half_dim < d:
        out = torch.cat([o0, o1, x[:, :, 2 * half_dim :]], dim=-1)
    else:
        out = torch.cat([o0, o1], dim=-1)

    return out


if __name__ == "__main__":

    torch.manual_seed(0)
    batch_size = 248
    seq_len = 100
    head_dim = 64
    batch_tokens = batch_size * seq_len
    x_shape = (batch_tokens, 32, 64)
    dtype = torch.float16
    q = torch.randn(x_shape, dtype=dtype, device="cuda")
    k = torch.clone(q)

    triton_q = q.view(batch_size, seq_len, 32, 64)
    triton_k = k.view(batch_size, seq_len, 32, 64)

    cos_shape = (batch_tokens, 32)
    y = torch.randn(cos_shape, dtype=dtype, device="cuda")
    cos = y.cos()
    sin = y.sin()

    triton_cos = cos.view(seq_len, 1, head_dim)
    triton_sin = sin.view(seq_len, 1, head_dim)

    output_torch = torch_rotary_emb(q, cos, sin)
    q_out, k_out, _, _ = rope_forward(triton_q, triton_k, triton_cos, triton_cos)
    triton_q_out = q_out.view(-1, 32, 64)
    print(
        f"output_torch shape {output_torch.shape}, triton_q_out shape {triton_q_out.shape}"
    )

    print(
        f"The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - triton_q_out))}"
    )
    print("torch:", triton.testing.do_bench(lambda: torch_rotary_emb(q, cos, sin)))
    print(
        "triton:",
        triton.testing.do_bench(lambda: rope_forward(triton_q, triton_k, cos, sin)),
    )
