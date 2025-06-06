import triton
import triton.language as tl
import torch
from triton.runtime.driver import CudaUtils


@triton.jit()
def _small_quantized_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales_g,
    stride_scales_n,
    stride_zeros_g,
    stride_zeros_n,
    groupsize,
    m,
    n,
    k,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    group_size_m: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
):

    pid = tl.program_id(0)

    total_blocks_m = tl.cdiv(m, block_size_m)
    total_blocks_n = tl.cdiv(n, block_size_n)
    total_blocks_k = tl.cdiv(k, block_size_k)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_m = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_n = pid_n * block_size_n + tl.arange(0, block_size_n)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_size_m), block_size_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_size_n), block_size_n)

    offs_k = tl.arange(0, block_size_k)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * block_size_m, 0),
        block_shape=(block_size_m, block_size_k),
        order=(1, 0),
    )

    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    output = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        g_id = k // (groupsize // block_size_k)

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        output += tl.dot(a, b)

        a_ptrs += stride_ak * block_size_k
        b_ptrs += (block_size_k // 8) * stride_bk

    output.to(tl.float16)
    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    tl.store(c_ptrs, output)


class small_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros):

        m, k = a.shape
        _, n = b.shape

        quant_groupsize = 256
        block_size_m = 16
        block_size_n = 32
        block_size_k = 512
        group_size_m = 8
        num_warps = 4
        num_stages = 8
        total_blocks_m = triton.cdiv(m, block_size_m)
        total_blocks_n = triton.cdiv(n, block_size_n)
        total_programs = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)
        fp8_fast_accum = False

        c = torch.zeros((m, n), device=b.device, dtype=torch.float16)

        k = _small_quantized_matmul[grid](
            a,
            b,
            c,
            scales,
            zeros,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            scales.stride(0),
            scales.stride(1),
            zeros.stride(0),
            zeros.stride(1),
            quant_groupsize,
            m,
            n,
            k,
            block_size_m,
            block_size_n,
            block_size_k,
            group_size_m,
            fp8_fast_accum=fp8_fast_accum,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c


small_qlinear = small_qlinear.apply


@triton.jit()
def _h100_quantized_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales_g,
    stride_scales_n,
    stride_zeros_g,
    stride_zeros_n,
    groupsize,
    m,
    n,
    k,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    group_size_m: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
):

    pid = tl.program_id(0)

    total_blocks_m = tl.cdiv(m, block_size_m)
    total_blocks_n = tl.cdiv(n, block_size_n)
    total_blocks_k = tl.cdiv(k, block_size_k)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_n = pid_n * block_size_n + tl.arange(0, block_size_n)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_size_n), block_size_n)
    offs_k = tl.arange(0, block_size_k)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * block_size_m, 0),
        block_shape=(block_size_m, block_size_k),
        order=(1, 0),
    )

    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    acc = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):

        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_ptrs)
        g_id = k // (groupsize // block_size_k)

        ptr = scales_ptrs + g_id * stride_scales_g

        scales = tl.load(ptr)
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        if fp8_fast_accum:
            acc = tl.dot(a.to(tl.float), b.to(tl.float8e4nv), acc)
        else:
            acc += tl.dot(a, b)

        a_block_ptr = tl.advance(a_block_ptr, (0, block_size_k))
        b_ptrs += (block_size_k // 8) * stride_bk

    acc.to(tl.float16)
    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < n) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, acc, mask=c_mask)


class h100_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros):

        m, k = a.shape
        _, n = b.shape

        quant_groupsize = 128
        block_size_m = 16
        block_size_n = 32
        block_size_k = 256
        group_size_m = 8
        num_warps = 4
        num_stages = 4
        total_blocks_m = triton.cdiv(m, block_size_m)
        total_blocks_n = triton.cdiv(n, block_size_n)
        total_programs = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)
        fp8_fast_accum = False

        c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
        k = _h100_quantized_matmul[grid](
            a,
            b,
            c,
            scales,
            zeros,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            scales.stride(0),
            scales.stride(1),
            zeros.stride(0),
            zeros.stride(1),
            quant_groupsize,
            m,
            n,
            k,
            block_size_m,
            block_size_n,
            block_size_k,
            group_size_m,
            fp8_fast_accum=fp8_fast_accum,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c


h100_qlinear = h100_qlinear.apply


@triton.jit
def matmul4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales_g,
    stride_scales_n,
    stride_zeros_g,
    stride_zeros_n,
    groupsize,
    NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
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
    a_mask = offs_am[:, None] < M

    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n

    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    if NO_GROUPS:

        scales = tl.load(scales_ptrs)
        zeros = tl.load(zeros_ptrs)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)

        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr = scales_ptrs + g_id * stride_scales_g

            scales = tl.load(ptr)
            ptr = zeros_ptrs + g_id * stride_zeros_g
            zeros = tl.load(ptr)

            zeros = (zeros >> zeros_shifter) & 0xF
            zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF

        b = b * scales[None, :] - zeros[None, :]

        b = b.to(tl.float16)

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul4(
    groupsize: int,
    a: torch.FloatTensor,
    qweight: torch.IntTensor,
    scales: torch.FloatTensor,
    qzeros: torch.IntTensor,
) -> torch.FloatTensor:

    assert a.shape[-1] == (
        qweight.shape[0] * 8
    ), "A must be a multiple of 8 in the last dimension"
    assert a.is_contiguous(), "A must be contiguous"

    x = a.view(-1, a.shape[-1])

    M, K = a.shape
    N = qweight.shape[1]

    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 32
    GROUP_SIZE_M = 8

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul4_kernel[grid](
        x,
        qweight,
        c,
        scales,
        qzeros,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        c.stride(0),
        c.stride(1),
        scales.stride(0),
        scales.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        groupsize,
        groupsize == K,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    c = c.view(a.shape[:-1] + (N,))

    return c


@triton.jit()
def matmul_split_k_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales_g,
    stride_scales_n,
    stride_zeros_g,
    stride_zeros_n,
    groupsize,
    m,
    n,
    k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    split_k: tl.constexpr,
):

    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    num_pid_k = tl.cdiv(k, block_k * split_k)

    pid_m, pid_n = swizzle_tile(pid, m, n, block_m, block_n, group_m)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_z * block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, num_pid_k):

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        g_id = k // (groupsize // block_k)

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        acc += tl.dot(a, b)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += (block_k // 8) * split_k * stride_bk

    acc.to(tl.float16)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc)


def matmul_split_k(a, b, scales, zeros):

    m, k = a.shape
    _, n = b.shape

    quant_groupsize = 128
    block_m = 16
    block_n = 32
    block_k = 256
    group_m = 8
    num_stages = 4
    num_warps = 4
    split_k = 2

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k

    grid = (total_programs_mn, total_programs_k)

    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    k = matmul_split_k_kernel[grid](
        a,
        b,
        c,
        scales,
        zeros,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        scales.stride(0),
        scales.stride(1),
        zeros.stride(0),
        zeros.stride(1),
        quant_groupsize,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        group_m,
        split_k,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return c


def make_tensor(M, N, dtype):
    if dtype == torch.int32:

        res = torch.randint(
            low=-2147483648, high=2147483647, size=(M, N), dtype=dtype, device="cuda"
        )
    else:

        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res


if __name__ == "__main__":

    m = 16
    k = 4096
    n = 4096
    groupsize = 128
    g = k // groupsize

    a = make_tensor(m, k, dtype=torch.float16)
    b = make_tensor(k // 8, n, dtype=torch.int32)
    c = make_tensor(m, n, dtype=torch.float16)
    zeros = make_tensor(g, n // 8, torch.int32)
    scales = make_tensor(g, n, torch.float16)

    small_output = small_qlinear(a, b, scales, zeros)
    splitk = matmul_split_k(a, b, scales, zeros)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n", "k"],
            x_vals=[256 * i for i in range(2, 25)],
            line_arg="provider",
            line_vals=["gptq", "custom"],
            line_names=["AutoGPTQ - Triton", "Custom Triton Kernel"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="gptq_vs_custom",
            args={},
        )
    )
    def benchmark(n, k, provider):

        m = 1
        groupsize = 128
        g = k // groupsize

        a = make_tensor(m, k, dtype=torch.float16)
        b = make_tensor(k // 8, n, dtype=torch.int32)

        zeros = make_tensor(g, n // 8, torch.int32)
        scales = make_tensor(g, n, torch.float16)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "gptq":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_matmul4(groupsize, a, b, scales, zeros),
                quantiles=quantiles,
            )
        if provider == "custom":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: h100_qlinear(a, b, scales, zeros), quantiles=quantiles
            )
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path="./")
