import math

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu_new(x):
    pi = tl.constexpr(tl.float32(math.pi))
    a = tl.math.sqrt(2.0 / pi)
    b = x + 0.044715 * x * x * x
    return 0.5 * x * (1.0 + tanh(a * b))


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _fused_linear_kernel_fwd(
    x_ptr,
    w_ptr,
    z_ptr,
    M,
    N,
    K,
    b_ptr=None,
    r_ptr=None,
    apply_silu=False,
    seed=1337,
    BLOCK_SIZE_M: tl.constexpr = 128,
    BLOCK_SIZE_N: tl.constexpr = 128,
    BLOCK_SIZE_K: tl.constexpr = 64,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]

    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k

        x = tl.load(x_ptr + offs_m * K + x_k, mask=(offs_m < M) & (x_k < K), other=0.0)
        x = x.to(tl.float16)

        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k

        w = tl.load(w_ptr + w_k * N + offs_n, mask=(w_k < K) & (offs_n < N), other=0.0)
        w = w.to(tl.float16)

        z = tl.dot(x, w, acc=z)

    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0)
        z += b.to(tl.float32)

    z_offset = offs_m * N + offs_n
    z_mask = (offs_m < M) & (offs_n < N)

    if apply_silu:
        z = silu(z)

    if r_ptr is not None:
        r = tl.load(r_ptr + z_offset, mask=z_mask)
        z += r.to(tl.float32)

    tl.store(z_ptr + z_offset, z, mask=z_mask)


@torch.no_grad()
def fused_linear(
    x,
    weight,
    bias=None,
    residual=None,
    add_silu=False,
):

    out_shape_0 = x.shape[:-1]

    x = x.view((-1, x.shape[-1]))
    M, K = x.shape
    N = weight.shape[1]

    z = torch.empty((M, N), device=x.device, dtype=x.dtype)

    assert x.is_contiguous()
    assert weight.is_contiguous()

    if bias is not None:
        assert bias.is_contiguous()
        assert weight.shape[1] == bias.shape[0]
    if residual is not None:
        residual = residual.view(z.shape)
        assert residual.is_contiguous()

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    _fused_linear_kernel_fwd[grid](
        x,
        weight,
        z,
        M,
        N,
        K,
        apply_silu=add_silu,
        b_ptr=bias,
        r_ptr=residual,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return z.view((*out_shape_0, N))
