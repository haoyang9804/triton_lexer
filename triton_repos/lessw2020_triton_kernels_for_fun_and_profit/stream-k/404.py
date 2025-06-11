import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
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
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, activation=""):

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


@triton.jit
def streamk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    scratchpad,
    locks,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
):
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_iters = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N) * iters_per_tile
    iters_per_cta = tl.cdiv(total_iters, tl.num_programs(0))
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    itr = pid * iters_per_cta
    iter_end = itr + iters_per_cta

    GROUP_SIZE_M = 8
    while itr < iter_end and itr < total_iters:
        tile_idx = itr // iters_per_tile
        tile_iter = tile_idx * iters_per_tile
        tile_iter_end = tile_iter + iters_per_tile

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_idx // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = tile_idx % num_pid_m
        pid_n = tile_idx // num_pid_m
        pid_k = itr - tile_iter

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        local_iter = pid_k
        local_iter_end = min(iter_end, tile_iter_end) - tile_iter

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, local_iter_end - local_iter):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        acc = acc.to(tl.float16)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        tile_started = itr == tile_iter
        tile_ended = iter_end >= tile_iter_end

        scratch_off = pid * BLOCK_SIZE_M * BLOCK_SIZE_N
        offs_scratch = (
            tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
            + tl.arange(0, BLOCK_SIZE_N)[None, :]
        )

        if USE_ATOMICS:
            tl.atomic_add(c_ptr + c_offs, acc, c_mask)
        else:
            if not tile_started:
                tl.store(scratchpad + scratch_off + offs_scratch, acc, c_mask)
                tl.atomic_xchg(locks + pid, 1)
            else:
                if not tile_ended:
                    cta_end = tl.cdiv(tile_iter_end, iters_per_cta)
                    cas = pid + 1
                    while cas < cta_end:
                        while tl.atomic_cas(locks + cas, 1, 2) != 1:
                            pass
                        acc += tl.load(
                            scratchpad
                            + cas * BLOCK_SIZE_M * BLOCK_SIZE_N
                            + offs_scratch,
                            c_mask,
                        )
                        cas += 1

                tl.store(c_ptr + c_offs, acc, c_mask)

        itr = tile_iter_end


torch.set_printoptions(sci_mode=False)


def streamk_matmul(a, b):
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64

    USE_ATOMICS = False

    if not USE_ATOMICS:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (108,)
    scratchpad = torch.empty(
        108, BLOCK_SIZE_M, BLOCK_SIZE_N, device=a.device, dtype=a.dtype
    )
    locks = torch.zeros(108, device="cuda", dtype=torch.int32)

    streamk_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        scratchpad,
        locks,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_ATOMICS=USE_ATOMICS,
        num_stages=3,
        num_warps=8,
    )
    return c


torch.set_printoptions(sci_mode=False)


from triton.testing import do_bench

torch.manual_seed(0)


def get_flops(M, N, K, fn=streamk_matmul):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    check_correct(A, B, fn)

    ms = do_bench(lambda: fn(A, B))
    print(f"{A.shape[0]}: {A.shape[0] * A.shape[1] * B.shape[1] * 2 / ms / 1e9}")


def check_correct(A, B, fn=streamk_matmul):
    out_triton = fn(A, B)
    out = torch.mm(A, B)
    torch.testing.assert_close(out, out_triton, atol=3e-1, rtol=1e-1)


import triton.ops as ops

for N in range(2048, 4096, 256):
    get_flops(N, N, N, ops.matmul)
    get_flops(N, N, N, streamk_matmul)
    print()
