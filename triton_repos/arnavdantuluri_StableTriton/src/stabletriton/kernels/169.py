from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx
from torch.cuda.amp import custom_fwd
from triton import JITFunction


def pytorch_naive_layernorm(
    a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
):

    mean = a.mean(dim=-1, keepdim=True)
    var = a.var(dim=-1, keepdim=True)
    rstd = 1 / torch.sqrt(var + eps)
    a_hat = (a - mean) * rstd
    out = a_hat * weight + bias
    return out


def pytorch_naive_rmsnorm(a: torch.Tensor, weight: torch.Tensor, eps: float):

    variance = a.to(torch.float32).pow(2).mean(-1, keepdim=True)
    a *= torch.rsqrt(variance + eps)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        a = a.to(weight.dtype)

    return weight * a


@triton.jit
def layer_norm_xformers(
    output_ptr,
    a_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    output_row_stride,
    output_col_stride,
    a_row_stride,
    a_col_stride,
    N_SIZE,
    eps,
    HAS_BIAS: tl.constexpr,
    IS_RMSNORM: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N_SIZE)
    mask = cols < N_SIZE

    x_ptrs = a_ptr + row * a_row_stride + cols * a_col_stride

    x = tl.load(x_ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(
        tl.float32
    )
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N_SIZE
    x_zm = tl.where(mask, x - mean, 0.0)
    tl.store(mean_ptr + row, mean)

    x_var = tl.sum(x_zm * x_zm, axis=0) / N_SIZE
    rstd = 1.0 / tl.sqrt(x_var + eps)

    y = x_zm * rstd
    tl.store(rstd_ptr + row, rstd)

    y = y * w + b
    y_ptrs = output_ptr + row * output_row_stride + cols * output_col_stride
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def _layer_norm_fwd_fused_single_pass(
    output_ptr,
    a_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    output_row_stride,
    output_col_stride,
    a_row_stride,
    a_col_stride,
    N_SIZE,
    eps,
    HAS_BIAS: tl.constexpr,
    IS_RMSNORM: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)

    a_row_off = row_idx * a_row_stride
    block_range_offs = tl.arange(0, BLOCK_N_SIZE)

    mean = 0.0
    var = 0.0
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        n_end_off = min((block_n_start_idx + BLOCK_N_SIZE), N_SIZE)
        block_cols_count = n_end_off - block_n_start_idx
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE

        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        if IS_RMSNORM:
            var += tl.sum(a * a, axis=0)
        else:
            block_mean = tl.sum(a, axis=0) / block_cols_count

            delta_mean = block_mean - mean
            delta_mean_sqr = delta_mean * delta_mean

            block_delta = tl.sum((a - block_mean) * a, axis=0)

            mean += tl.sum((a - mean) * a_ptr_mask, axis=0) / n_end_off
            var += (
                block_delta
                + delta_mean_sqr * (block_n_start_idx * block_cols_count) / n_end_off
            )

    var /= N_SIZE
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)

    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        weight = tl.load(weight_ptr + col_offs, mask=a_ptr_mask)

        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + col_offs, mask=a_ptr_mask)
            out = out + bias

        tl.store(
            output_ptr + row_idx * output_row_stride + col_offs * output_col_stride,
            out,
            mask=a_ptr_mask,
        )


@triton.jit
def _layer_norm_fwd_fused_multi_pass(
    output_ptr,
    a_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    output_row_stride,
    output_col_stride,
    a_row_stride,
    a_col_stride,
    N_SIZE,
    eps,
    IS_RMSNORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    row_off = row_idx * a_row_stride
    block_range_offs = tl.arange(0, BLOCK_N_SIZE)

    mean_acc = tl.zeros((BLOCK_N_SIZE,), dtype=tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        cols_offs = block_n_start_idx + block_range_offs
        a = tl.load(
            a_ptr + row_off + cols_offs * a_col_stride,
            mask=cols_offs < N_SIZE,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        mean_acc += a
    mean = tl.sum(mean_acc, axis=0) / N_SIZE

    var_acc = tl.zeros((BLOCK_N_SIZE,), dtype=tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        cols_offs = block_n_start_idx + block_range_offs
        a = tl.load(
            a_ptr + row_off + cols_offs * a_col_stride,
            mask=cols_offs < N_SIZE,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        a = tl.where(cols_offs < N_SIZE, a - mean, 0.0)
        var_acc += a * a
    var = tl.sum(var_acc, axis=0) / N_SIZE

    rstd = 1 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)

    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        cols_offs = block_n_start_idx + tl.arange(0, BLOCK_N_SIZE)
        mask_ptr = cols_offs < N_SIZE
        weight = tl.load(weight_ptr + cols_offs, mask=mask_ptr)
        bias = tl.load(bias_ptr + cols_offs, mask=mask_ptr)
        a = tl.load(
            a_ptr + row_off + cols_offs * a_col_stride,
            mask=mask_ptr,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        a_hat = (a - mean) * rstd
        output = a_hat * weight + bias

        tl.store(
            output_ptr + row_idx * output_row_stride + cols_offs * output_col_stride,
            output,
            mask=mask_ptr,
        )


class LayerNorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        implementation: JITFunction,
        use_rms_norm: bool,
    ):
        assert (
            x.dtype == weight.dtype
        ), f"input and weight bias must have the same dtype: {x.dtype}, {weight.dtype}"
        if bias is not None:
            assert (
                x.dtype == bias.dtype
            ), f"input and bias must have the same dtype: {x.dtype}, {bias.dtype}"

        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        out = torch.empty_like(x)

        a_arg = x.reshape(-1, x.shape[-1])
        M, N = a_arg.shape

        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        std = torch.empty((M,), dtype=torch.float32, device="cuda")

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        BLOCK_SIZE = max(BLOCK_SIZE, 128)
        if implementation == layer_norm_xformers:
            assert N <= 4096, "LayerNorm: N is too large for xformers implementation"
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)

        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        implementation[(M,)](
            output_ptr=out,
            a_ptr=a_arg,
            weight_ptr=weight,
            bias_ptr=bias if bias is not None else a_arg,
            mean_ptr=mean,
            rstd_ptr=std,
            output_row_stride=out.stride(-2),
            output_col_stride=out.stride(-1),
            a_row_stride=a_arg.stride(0),
            a_col_stride=a_arg.stride(1),
            N_SIZE=N,
            eps=eps,
            HAS_BIAS=bias is not None,
            IS_RMSNORM=use_rms_norm,
            BLOCK_N_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, mean, std, weight)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return out


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    implementation: JITFunction = _layer_norm_fwd_fused_single_pass,
    use_rms_norm: bool = False,
):
    return LayerNorm.apply(x, weight, bias, eps, implementation, use_rms_norm)


def test_layer_norm(M, N, dtype, eps=1e-5, device="cuda"):

    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    y = torch.nn.LayerNorm((5, 5), eps=1e-05, elementwise_affine=True).cuda()
    weight = y.weight
    bias = y.weight
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    x.requires_grad_(True)

    y_tri = layer_norm(x, weight, bias, eps)
    y_ref = y(x)

    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


if __name__ == "__main__":
    M, N = 5, 5
    test_layer_norm(M, N, torch.float32)
