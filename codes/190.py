import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n: int) -> (
    int,
    int,
):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def layernorm_forward(
    Y,
    Y_row_stride,
    X,
    X_row_stride,
    weight,
    bias,
    inv_var,
    mean,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)

    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride

    inv_var += row_idx
    mean += row_idx

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)

    weight_row = tl.load(weight + col_offsets, mask=mask, other=0).to(tl.float32)

    bias_row = tl.load(bias + col_offsets, mask=mask, other=0).to(tl.float32)

    mean_X = tl.sum(X_row, axis=0) / n_cols

    XX = tl.where(mask, X_row - mean_X, 0)

    row_var = tl.sum(XX * XX, axis=0) / n_cols

    inv_var_val = tl.math.rsqrt(row_var + eps)

    tl.store(inv_var, inv_var_val)
    tl.store(mean, mean_X)

    output = (XX * inv_var_val) * weight_row + bias_row

    tl.store(Y + col_offsets, output, mask=mask)


@triton.jit
def layernorm_backward(
    dY,
    dY_row_stride,
    X,
    X_row_stride,
    weight,
    bias,
    inv_var,
    mean,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)

    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    inv_var += row_idx
    mean += row_idx

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)

    weight_row = tl.load(weight + col_offsets, mask=mask, other=0).to(tl.float32)

    inv_var_val = tl.load(inv_var).to(tl.float32)
    mean_val = tl.load(mean).to(tl.float32)

    normed = (X_row - mean_val) * inv_var_val

    dY_W = dY_row * weight_row

    dX_row = (
        dY_W
        - tl.sum(dY_W, axis=0) / n_cols
        - normed * tl.sum(dY_W * normed, axis=0) / n_cols
    )
    dX_row = dX_row * inv_var_val

    tl.store(dY + col_offsets, dX_row, mask=mask)


class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device
        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        inv_var = torch.empty(n_rows, dtype=torch.float32, device=device)
        mean = torch.empty(n_rows, dtype=torch.float32, device=device)

        layernorm_forward[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            weight,
            bias,
            inv_var,
            mean,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, weight, bias, inv_var, mean)
        return Y.view(*shape)

    pass

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, weight, bias, inv_var, mean = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        layernorm_backward[(n_rows,)](
            dY,
            dY.stride(0),
            X,
            X.stride(0),
            weight,
            bias,
            inv_var,
            mean,
            n_cols,
            ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None, None


def fast_layernorm(layernorm, X):
    assert layernorm.elementwise_affine is True
    W = layernorm.weight
    bias = layernorm.bias
    eps = (
        layernorm.variance_epsilon
        if hasattr(layernorm, "variance_epsilon")
        else layernorm.eps
    )
    out = Fast_Layernorm.apply(X, W, bias, eps)
    return out
