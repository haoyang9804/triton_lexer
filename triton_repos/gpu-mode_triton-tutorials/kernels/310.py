import torch
import triton
import triton.language as tl


@triton.jit
def outerk_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):

        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)

        mask_k = offsets_k < K

        a_ptrs = a_ptr + (
            offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        )

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_outer_k_matmul(a, b):

    assert a.is_cuda and b.is_cuda, "a and b must be on GPU"
    assert a.shape[1] == b.shape[0], "mismatch between inner dimensions"

    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    outerk_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

    return c


def benchmark_matmul():

    M, N, K = 8192, 8192, 4096
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    torch_output = torch.matmul(a, b)
    triton_output = triton_outer_k_matmul(a, b)
    assert torch.allclose(
        torch_output, triton_output, rtol=1e-2, atol=1e-1
    ), "Triton and PyTorch matmul results don't match!"

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.matmul(a, b)
    start.record()
    for _ in range(10):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / 10

    triton_outer_k_matmul(a, b)
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        triton_outer_k_matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / 10

    print(f"PyTorch matmul time: {pytorch_time:.2f} ms")
    print(f"Triton matmul time: {triton_time:.2f} ms")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")


if __name__ == "__main__":
    benchmark_matmul()
