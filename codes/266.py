from typing import Optional
from einops import einsum
import torch
import triton
import triton.language as tl
from native_sparse_attention.ops.triton.utils import get_compressed_seqlens


@triton.jit
def sliding_pool_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    cu_seqlens,
    y_cu_seqlens,
    head_dim,
    kernel_size,
    kernel_stride,
    stride_xn,
    stride_xh,
    stride_xd,
    stride_yn,
    stride_yh,
    stride_yd,
    stride_wh,
    stride_wk,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    x_start = tl.load(cu_seqlens + pid_b)
    x_len = tl.load(cu_seqlens + pid_b + 1) - x_start
    y_start = tl.load(y_cu_seqlens + pid_b)
    y_len = tl.load(y_cu_seqlens + pid_b + 1) - y_start
    if pid_k >= y_len:
        return
    if w_ptr is not None:

        w_ptrs = tl.make_block_ptr(
            base=w_ptr + pid_h * stride_wh,
            shape=(kernel_size, 1),
            strides=(stride_wk, 0),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_K, 1),
            order=(0, 1),
        )
        w = tl.load(w_ptrs, boundary_check=(0, 1), padding_option="zero")

    x_ptrs = tl.make_block_ptr(
        base=x_ptr + x_start * stride_xn + pid_h * stride_xh,
        shape=(x_len, head_dim),
        strides=(stride_xn, stride_xd),
        offsets=(pid_k * kernel_stride, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")

    if w_ptr is not None:
        y = tl.sum(x * w, axis=0)
    else:
        y = tl.sum(x, axis=0) / kernel_size
    off_d = tl.arange(0, BLOCK_SIZE_D)
    tl.store(
        y_ptr + (y_start + pid_k) * stride_yn + pid_h * stride_yh + off_d * stride_yd,
        y.to(y_ptr.dtype.element_ty),
        mask=off_d < head_dim,
    )


@triton.jit
def sliding_pool_dxdw_kernel(
    x_ptr,
    dx_ptr,
    dy_ptr,
    w_ptr,
    dw_ptr,
    cu_seqlens,
    y_cu_seqlens,
    head_dim,
    kernel_size,
    kernel_stride,
    stride_xn,
    stride_xh,
    stride_xd,
    stride_dxn,
    stride_dxh,
    stride_dxd,
    stride_dyn,
    stride_dyh,
    stride_dyd,
    stride_wh,
    stride_wk,
    stride_dwh,
    stride_dwn,
    stride_dwk,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    x_start = tl.load(cu_seqlens + pid_b)
    x_len = tl.load(cu_seqlens + pid_b + 1) - x_start
    y_start = tl.load(y_cu_seqlens + pid_b)
    y_len = tl.load(y_cu_seqlens + pid_b + 1) - y_start
    if pid_k >= y_len:
        return

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    if w_ptr is not None:

        w_ptrs = w_ptr + pid_h * stride_wh + off_k * stride_wk
        w = tl.load(w_ptrs, mask=off_k < kernel_size, other=0)

    x_ptrs = tl.make_block_ptr(
        base=x_ptr + x_start * stride_xn + pid_h * stride_xh,
        shape=(head_dim, x_len),
        strides=(stride_xd, stride_xn),
        offsets=(0, pid_k * kernel_stride),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero")

    dy_ptrs = (
        dy_ptr
        + pid_h * stride_dyh
        + (y_start + pid_k) * stride_dyn
        + off_d * stride_dyd
    )
    dy = tl.load(dy_ptrs, mask=off_d < head_dim, other=0)
    if w_ptr is not None:

        dx = dy[None, :] * w[:, None]

        dw = tl.sum(dy[:, None] * x, axis=0)

        dw_ptrs = (
            dw_ptr
            + pid_h * stride_dwh
            + (y_start + pid_k) * stride_dwn
            + off_k * stride_dwk
        )
        tl.store(dw_ptrs, dw.to(dw_ptr.dtype.element_ty), mask=off_k < kernel_size)
    else:
        dx = dy[None, :] / kernel_size

    dx_ptrs = (
        dx_ptr
        + pid_h * stride_dxh
        + (x_start + pid_k * kernel_stride + off_k[:, None]) * stride_dxn
        + off_d[None, :] * stride_dxd
    )
    tl.atomic_add(
        dx_ptrs,
        dx.to(dx_ptr.dtype.element_ty),
        mask=(off_k < x_len - pid_k * kernel_stride)[:, None]
        & (off_d < head_dim)[None, :],
    )


class SlidingWindowWeightedPool(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kernel_size: int,
        kernel_stride: int,
    ):

        assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
        if w is not None:
            assert x.dtype == w.dtype
        assert cu_seqlens.dtype == torch.int32

        total_len, num_heads, head_dim = x.shape
        batch_size = cu_seqlens.shape[0] - 1
        if w is not None:
            assert w.shape[0] == num_heads
            assert w.shape[1] == kernel_size
        assert kernel_size % kernel_stride == 0
        assert kernel_size in {16, 32, 64, 128}

        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        y_seqlens, y_cu_seqlens = get_compressed_seqlens(
            cu_seqlens, kernel_size, kernel_stride
        )

        y = torch.zeros(
            y_cu_seqlens[-1], num_heads, head_dim, dtype=x.dtype, device=x.device
        )

        BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
        BLOCK_SIZE_K = triton.next_power_of_2(kernel_size)
        grid = (batch_size, num_heads, y_seqlens.max().item())
        sliding_pool_fwd_kernel[grid](
            x,
            y,
            w,
            cu_seqlens,
            y_cu_seqlens,
            head_dim,
            kernel_size,
            kernel_stride,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            w.stride(0) if w is not None else None,
            w.stride(1) if w is not None else None,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
        ctx.save_for_backward(x, w, seqlens, cu_seqlens, y_seqlens, y_cu_seqlens)
        ctx.kernel_size = kernel_size
        ctx.kernel_stride = kernel_stride
        ctx.head_dim = head_dim
        return y, y_cu_seqlens

    @staticmethod
    def backward(ctx, dy, _):
        x, w, seqlens, cu_seqlens, y_seqlens, y_cu_seqlens = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        kernel_stride = ctx.kernel_stride
        head_dim = ctx.head_dim
        batch_size = cu_seqlens.shape[0] - 1
        num_heads = x.shape[1]

        dx = torch.zeros_like(x, dtype=torch.float32)
        if w is not None:
            dw = torch.zeros(
                num_heads,
                y_cu_seqlens[-1],
                kernel_size,
                dtype=torch.float32,
                device=w.device,
            )
        BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
        BLOCK_SIZE_K = triton.next_power_of_2(kernel_size)
        grid = (batch_size, num_heads, y_seqlens.max().item())
        sliding_pool_dxdw_kernel[grid](
            x,
            dx,
            dy,
            w,
            dw if w is not None else None,
            cu_seqlens,
            y_cu_seqlens,
            head_dim,
            kernel_size,
            kernel_stride,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dy.stride(0),
            dy.stride(1),
            dy.stride(2),
            w.stride(0) if w is not None else None,
            w.stride(1) if w is not None else None,
            dw.stride(0) if w is not None else None,
            dw.stride(1) if w is not None else None,
            dw.stride(2) if w is not None else None,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
        dx = dx.to(x.dtype)
        if w is None:
            dw = None
        else:
            dw = dw.sum(1).to(w.dtype)
        return dx, dw, None, None, None


def weightedpool_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):

    y, y_cu_seqlens = SlidingWindowWeightedPool.apply(
        x, w, cu_seqlens, kernel_size, kernel_stride
    )

    if pe is not None:
        assert pe.dtype == x.dtype and pe.device == x.device
        bias = einsum(pe, w, "h k d, h k -> h d")
        y = y + bias.unsqueeze(0)
    return y, y_cu_seqlens


def avgpool_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):

    assert w is None, "don't need additional weight for avgpool"
    y, y_cu_seqlens = SlidingWindowWeightedPool.apply(
        x, w, cu_seqlens, kernel_size, kernel_stride
    )

    if pe is not None:
        assert pe.dtype == x.dtype and pe.device == x.device
        bias = torch.mean(pe, dim=1)
        y = y + bias.unsqueeze(0)
    return y, y_cu_seqlens


if __name__ == "__main__":
    from native_sparse_attention.ops.torch.compress_key_value import (
        weightedpool_compress_torch,
    )

    torch.manual_seed(42)
    num_heads = 4
    head_dim = 128
    kernel_size = 32
    kernel_stride = 16
    seqlens = torch.LongTensor([12, 1000, 2000, 4096]).int().cuda()
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    x = (
        torch.zeros(cu_seqlens[-1], num_heads, head_dim)
        .uniform_(-1, 1)
        .to(torch.bfloat16)
        .cuda()
        .requires_grad_()
    )
    w = (
        torch.zeros(num_heads, kernel_size)
        .uniform_(-1 / 32, 1 / 32)
        .cuda()
        .to(torch.bfloat16)
        .requires_grad_()
    )

    y, y_cu_seqlens = weightedpool_compress_torch(
        x, w, cu_seqlens, kernel_size, kernel_stride, None
    )

    x1 = x.clone().detach().requires_grad_()
    w1 = w.clone().detach().requires_grad_()

    y1, y1_cu_seqlens = weightedpool_compress(
        x1, w1, cu_seqlens, kernel_size, kernel_stride
    )

    print(torch.allclose(y, y1, rtol=1e-2, atol=1e-2))
    print(torch.abs(y - y1).max().item())

    randn = torch.randn_like(y)
    randn1 = randn.clone().detach()

    loss = (y * randn).sum()
    loss1 = (y1 * randn1).sum()
    loss.backward()
    loss1.backward()

    print((x.grad - x1.grad).abs().max())
    print(((w.grad - w1.grad).abs() / w.grad.abs()).max())
