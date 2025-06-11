import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):

    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w

        tl.store(Y + cols, y.to(tl.float16), mask=mask)


def rmsnorm_forward(x, weight, eps):

    y = torch.empty_like(x)

    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    BLOCK_SIZE = 128 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    num_warps = 8

    _rms_norm_fwd_fused[(M,)](
        x_arg,
        y,
        weight,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


import torch


def test_rmsnorm_forward():
    results = {}

    x1 = torch.randn(2, 64, dtype=torch.float16).cuda()
    weight1 = torch.randn(64, dtype=torch.float16).cuda()
    eps1 = 1e-5
    y1 = rmsnorm_forward(x1, weight1, eps1)
    results["test_case_1"] = y1

    x2 = torch.randn(4, 128, dtype=torch.float16).cuda()
    weight2 = torch.randn(128, dtype=torch.float16).cuda()
    eps2 = 1e-5
    y2 = rmsnorm_forward(x2, weight2, eps2)
    results["test_case_2"] = y2

    x3 = torch.randn(1, 8192, dtype=torch.float16).cuda()
    weight3 = torch.randn(8192, dtype=torch.float16).cuda()
    eps3 = 1e-5
    y3 = rmsnorm_forward(x3, weight3, eps3)
    results["test_case_3"] = y3

    x4 = torch.randn(1, 1, dtype=torch.float16).cuda()
    weight4 = torch.randn(1, dtype=torch.float16).cuda()
    eps4 = 1e-5
    y4 = rmsnorm_forward(x4, weight4, eps4)
    results["test_case_4"] = y4

    return results


result_gold = test_rmsnorm_forward()
