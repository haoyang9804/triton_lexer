import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,
    Y,
    stride_x_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):

    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row

    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)

    mask = cols < N
    y = x * rstd

    tl.store(Y + cols, y, mask=mask)


@triton.jit
def _l2_norm_bwd_kernel(
    X,
    DY,
    DX,
    stride_x_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):

    row = tl.program_id(0)
    X += row * stride_x_row
    DX += row * stride_x_row
    DY += row * stride_x_row

    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)

    mask = cols < N
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)


def _l2_norm_fwd(x, eps=1e-6):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    assert x.stride(-1) == 1

    y = torch.empty_like(x)
    assert y.stride(-1) == 1
    N = x.shape[-1]
    M = x.shape[0]

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    with torch.cuda.device(x.device.index):
        _l2_norm_fwd_1pass_kernel[(M,)](
            x,
            y,
            x.stride(0),
            N,
            eps,
            BLOCK_N,
        )
    return y.reshape(x_shape_og)


def _l2_norm_bwd(
    x,
    dy,
    eps=1e-5,
):
    x_shape_og = x.shape
    x = x.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    assert dy.shape == x.shape

    dx = torch.empty_like(x)
    N = x.shape[-1]
    M = x.shape[0]
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    with torch.cuda.device(x.device.index):
        _l2_norm_bwd_kernel[(M,)](
            x,
            dy,
            dx,
            x.stride(0),
            N,
            eps,
            BLOCK_N,
        )
    return dx.reshape(x_shape_og)


import torch


def test_l2_norm_triton():

    batch_size, dim = 8, 128
    eps = 1e-6

    x = torch.randn(
        (batch_size, dim), dtype=torch.float32, device="cuda", requires_grad=True
    )

    test_results = {}

    y = _l2_norm_fwd(x, eps=eps)
    test_results["test_case_1"] = y

    dy = torch.ones_like(y, device="cuda")
    dx_analytical = _l2_norm_bwd(x, dy, eps=eps)
    test_results["test_case_2"] = dx_analytical

    return test_results


result_gold = test_l2_norm_triton()
