import os
import time
import torch
import torch.nn as nn

import triton
import triton.language as tl
import triton_dejavu


@triton_dejavu.autotune(
    configs=[
        triton.Config({"BLOCK_N_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config(
            {"BLOCK_N_SIZE": 1024},
            num_warps=8,
            num_stages=4,
            pre_hook=lambda nargs: nargs["output_ptr"].zero_(),
        ),
        triton.Config({"BLOCK_N_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config(
            {"BLOCK_N_SIZE": 4096},
            num_warps=4,
            num_stages=2,
            pre_hook=lambda nargs: nargs["output_ptr"].zero_(),
        ),
    ],
    rep=10,
    warmup=5,
    key=[
        "stride_x_batch",
        "stride_x_m",
        "stride_x_k",
        "stride_rms_w",
        "stride_out_batch",
        "stride_out_m",
        "stride_out_k",
        "N_SIZE",
    ],
    use_cuda_graph=True,
)
@triton.jit
def rmsnorm_triton(
    x_ptr,
    rms_w_ptr,
    output_ptr,
    stride_x_batch,
    stride_x_m,
    stride_x_k,
    stride_rms_w,
    stride_out_batch,
    stride_out_m,
    stride_out_k,
    N_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += x.to(tl.float32) * x.to(tl.float32)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = 1 / tl.math.sqrt(var + eps)

    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(
            x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0
        ).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = (
            pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        )
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    out = torch.empty_like(x)
    if len(x.shape) == 3:
        batch, M, K = x.shape
        stride_x_batch, stride_x_m, stride_x_k = x.stride()
        stride_rms_w = rms_w.stride()[0]
        stride_out_batch, stride_out_m, stride_out_k = out.stride()
    else:
        batch, K = x.shape
        M = 1
        stride_x_batch, stride_x_k = x.stride()
        stride_x_m = 1
        stride_rms_w = rms_w.stride()[0]
        stride_out_batch, stride_out_k = out.stride()
        stride_out_m = 1
    assert rms_w.shape[-1] == K

    rmsnorm_triton[
        (
            batch,
            M,
        )
    ](
        x,
        rms_w,
        out,
        stride_x_batch,
        stride_x_m,
        stride_x_k,
        stride_rms_w,
        stride_out_batch,
        stride_out_m,
        stride_out_k,
        eps=eps,
        N_SIZE=K,
    )
    return out
