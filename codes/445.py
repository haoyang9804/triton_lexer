import torch
import triton
import triton.language as tl
from .utils import calculate_settings


@triton.jit
def skip_rms_norm_kernel_no_view(
    Y_ptr,
    X_ptr,
    R_ptr,
    W_ptr,
    B,
    S,
    N,
    x_stride_b,
    x_stride_s,
    x_stride_n,
    r_stride_b,
    r_stride_s,
    r_stride_n,
    y_stride_b,
    y_stride_s,
    y_stride_n,
    w_stride,
    eps,
    has_residual: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    X_ptr = X_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s
    Y_ptr = Y_ptr + batch_idx * y_stride_b + seq_idx * y_stride_s

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + cols * x_stride_n, mask=mask, other=0.0).to(tl.float32)

    if has_residual:
        R_ptr = R_ptr + batch_idx * r_stride_b + seq_idx * r_stride_s
        r = tl.load(R_ptr + cols * r_stride_n, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(R_ptr + cols * r_stride_n, x, mask=mask)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + cols * w_stride, mask=mask, other=0.0)
    y = (x * rrms).to(tl.float16) * w

    tl.store(Y_ptr + cols * y_stride_n, y, mask=mask)


@torch.no_grad()
def skip_rmsnorm_no_view(X, residual, weight, eps=1e-5):

    B, S, N = X.shape
    Y = torch.empty_like(X)

    x_stride_b, x_stride_s, x_stride_n = X.stride()
    y_stride_b, y_stride_s, y_stride_n = Y.stride()
    w_stride = weight.stride(0)

    if residual is not None:
        residual = residual.contiguous()
        r_stride_b, r_stride_s, r_stride_n = residual.stride()
        has_residual = True
    else:

        r_stride_b, r_stride_s, r_stride_n = 0, 0, 0
        has_residual = False

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (B * S,)

    skip_rms_norm_kernel_no_view[grid](
        Y,
        X,
        residual if residual is not None else X,
        weight,
        B,
        S,
        N,
        x_stride_b,
        x_stride_s,
        x_stride_n,
        r_stride_b,
        r_stride_s,
        r_stride_n,
        y_stride_b,
        y_stride_s,
        y_stride_n,
        w_stride,
        eps,
        has_residual=has_residual,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (Y, residual) if residual is not None else (Y, X)


@triton.jit()
def rms_norm_kernel(
    Y,
    X,
    W,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@triton.jit()
def skip_rms_norm_kernel(
    Y,
    X,
    R,
    W,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r
    tl.store(R + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@torch.no_grad()
def skip_rmsnorm(X, residual, weight, eps=1e-5):
    orig_shape = X.shape
    X = X.view(-1, orig_shape[-1])

    M, N = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(N)
    Y = torch.empty_like(X)

    if residual is not None:
        residual = residual.view(-1, N)
        skip_rms_norm_kernel[M,](
            Y,
            X,
            residual,
            weight,
            N,
            1,
            N,
            1,
            N,
            1,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return Y.view(orig_shape), residual.view(orig_shape)
    else:
        rms_norm_kernel[M,](
            Y,
            X,
            weight,
            N,
            1,
            N,
            1,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return Y.view(orig_shape), X.view(orig_shape)


import pytest
import time


def python_rmsnorm(x, w, eps=1e-5):

    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x / torch.sqrt(var + eps)
    return x_normed * w


def python_skip_rmsnorm(x, r, w, eps=1e-5):

    x = x + r
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x / torch.sqrt(var + eps)
    return (x_normed * w).half(), x.half()


@pytest.mark.parametrize(
    "batch_size, N, hidden_size", [(4, 128, 4096), (2, 256, 4096), (8, 1024, 4096)]
)
def test_rmsnorm(batch_size, N, hidden_size):
    x = torch.randn(batch_size, N, hidden_size, device="cuda", dtype=torch.float16)
    w = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    y_ref = python_rmsnorm(x.float(), w.float()).half()
    y_triton, triton_residual = skip_rmsnorm(x, None, w)

    assert torch.allclose(
        y_ref, y_triton, atol=1e-3, rtol=1e-3
    ), "RMSNorm results do not match"


@pytest.mark.parametrize(
    "batch_size, N, hidden_size", [(4, 128, 4096), (2, 256, 4096), (8, 1024, 4096)]
)
def test_skip_rmsnorm(batch_size, N, hidden_size):
    x = torch.randn(batch_size, N, hidden_size, device="cuda", dtype=torch.float16)
    r = torch.randn(batch_size, N, hidden_size, device="cuda", dtype=torch.float16)
    w = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    y_ref, py_residual = python_skip_rmsnorm(x.float(), r.float(), w.float())
    y_triton, triton_residual = skip_rmsnorm(x, r, w)

    assert torch.allclose(
        y_ref, y_triton, atol=1e-3, rtol=1e-3
    ), "Skip RMSNorm results do not match"
    assert torch.allclose(
        py_residual, triton_residual, atol=1e-3, rtol=1e-3
    ), "Skip RMSNorm residual results do not match"


def benchmark_skip_rmsnorm(batch_size, N, iters=1000):
    x = torch.randn(batch_size, N, device="cuda", dtype=torch.float16)
    r = torch.randn(batch_size, N, device="cuda", dtype=torch.float16)
    w = torch.randn(N, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        skip_rmsnorm(x, r, w)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / iters
    print(f"skip_rmsnorm: B={batch_size}, N={N}, avg_time={avg_time * 1e3:.3f} ms/iter")


def benchmark(func, shapes, warmup=10, iters=50):
    times = []
    for shape in shapes:
        X = torch.randn(shape, dtype=torch.float16, device="cuda")
        R = torch.randn(shape, device="cuda", dtype=torch.float16)
        W = torch.randn(shape[-1], dtype=torch.float16, device="cuda")

        for _ in range(warmup):
            _ = func(X, R, W)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = func(X, R, W)
        torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / iters
        times.append(avg_time)
    return times


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    shapes = [(16, 2048, 4096), (32, 2048, 4096), (64, 2048, 4096), (256, 2048, 4096)]
    original_times = benchmark(skip_rmsnorm, shapes)
    optimized_times = benchmark(skip_rmsnorm_no_view, shapes)

    plt.figure(figsize=(8, 5))
    x_axis = [s[0] * s[1] for s in shapes]
    plt.plot(x_axis, original_times, color="red", label="Original")
    plt.plot(x_axis, optimized_times, color="blue", label="Optimized")
    plt.xlabel("Batch * Seq (M dimension)")
    plt.ylabel("Time (s)")
    plt.title("RMSNorm Kernel Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("./skip_rmsnorm_benchmark.png")
