import triton, torch
import triton.language as tl


@triton.jit
def _layernorm_kernel_fwd(
    x_ptr,
    weight_ptr,
    bias_ptr,
    z_ptr,
    H,
    eps=1e-5,
    BLOCK_SIZE: tl.constexpr = 16,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * H
    z_row_ptr = z_ptr + row_idx * H

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, H, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row_ptr + col_offsets, mask=col_offsets < H)
        _sum += x.to(tl.float32)

    mean = tl.sum(_sum, axis=0) / H

    x_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, H, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row_ptr + col_offsets, mask=col_offsets < H).to(tl.float32)
        x = tl.where(col_offsets < H, x - mean, 0.0)
        x_var += x * x

    x_var = tl.sum(x_var, axis=0) / H
    rtsd = tl.sqrt(x_var + eps)

    for i in range(0, H, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < H
        x = tl.load(x_row_ptr + col_offsets, mask=mask)
        w = tl.load(weight_ptr + col_offsets, mask=mask)
        b = tl.load(bias_ptr + col_offsets)

        x_hat = (x - mean) / rtsd
        z = x_hat * w + b
        tl.store(z_row_ptr + col_offsets, z, mask=mask)


@torch.no_grad()
def layernorm(x, weight, bias, eps=1e-5):

    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()

    assert x.shape[-1] == weight.shape[0] == bias.shape[0]
    out_shape = x.shape
    x = x.view(-1, x.shape[-1])
    BL, H = x.shape
    z = torch.empty(x.shape, device=x.device, dtype=x.dtype)

    MAX_FUSED_SIZE = 4096 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _layernorm_kernel_fwd[BL,](
        x, weight, bias, z, H, eps, BLOCK_SIZE, num_warps=num_warps
    )
    return z.view(out_shape)
