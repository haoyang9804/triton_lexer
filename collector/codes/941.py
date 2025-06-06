import operator

import torch
import triton
import triton.language as tl
from flux_triton.ops.utils import (
    calculate_settings,
    compare_version,
    ensure_contiguous,
)

if compare_version("triton", operator.ge, "3.0.0"):
    try:

        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:

        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


_CASTING_MODE_NONE = tl.constexpr(-1)
_CASTING_MODE_LLAMA = tl.constexpr(0)
_CASTING_MODE_GEMMA = tl.constexpr(1)


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    tl.store(RSTD_ptr, rstd)

    X_row = X_row * rstd

    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    Y_row = X_row * (offset + W_row)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_ptr += row_idx * dY_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
    dW_ptr += row_idx * dW_row_stride

    dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0)
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    original_x_dtype = X_row.dtype

    rstd_row = tl.load(RSTD_ptr)

    W_row = W_row + offset

    X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_LLAMA:
        m = (dY_row * W_row).to(tl.float32)

    elif casting_mode == _CASTING_MODE_GEMMA:
        dY_row, W_row = (
            dY_row.to(tl.float32),
            W_row.to(tl.float32),
        )

    m = dY_row * W_row

    dX_row = rstd_row * m

    dX_row += (rstd_row) * (
        -(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
    )

    if casting_mode == _CASTING_MODE_LLAMA:
        dW_row = dY_row * (X_row * rstd_row).to(original_x_dtype)
    else:

        dW_row = dY_row * (X_row * rstd_row)

    tl.store(dY_ptr + col_offsets, dX_row, mask=mask)
    tl.store(dW_ptr + col_offsets, dW_row, mask=mask)


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def rms_norm_forward(X, W, eps, offset, casting_mode):
    if not isinstance(casting_mode, int):
        assert (
            casting_mode in _str_to_casting_mode
        ), f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert (
            casting_mode in _str_to_casting_mode.values()
        ), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    rstd_dtype = (
        torch.float32
        if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value)
        else X.dtype
    )
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    assert (
        X.shape[1] == W.shape[0]
    ), "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"

    _rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        offset,
        casting_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape
    dW = torch.empty_like(
        X,
        dtype=(torch.float32 if casting_mode == _CASTING_MODE_GEMMA.value else W.dtype),
    )

    _rms_norm_backward_kernel[(n_rows,)](
        dY,
        dY.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        dW,
        dW.stride(0),
        n_cols,
        offset,
        casting_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dX = dY.view(*shape)
    dW = torch.sum(dW, dim=0).to(W.dtype)
    return dX, dW


class LigerRMSNormFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama"):

        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(
            X, W, eps, offset, casting_mode
        )
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):

        X, W, RSTD = ctx.saved_tensors
        dX, dW = rms_norm_backward(
            dY,
            X,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
        )
        return dX, dW, None, None, None
