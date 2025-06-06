import torch, time
import triton
import triton.language as tl


@triton.jit
def dequantize(
    b,
    scales,
    zeros,
    q_shift,
    meta_dtype,
    unpack_mask,
    elements_per_sample: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
):

    if elements_per_sample > 1:

        b = (b >> q_shift) & unpack_mask

    if W_group_mode == 1:

        b = b.to(meta_dtype) - zeros

    if W_group_mode == 2:

        b = b.to(meta_dtype) * scales

    if W_group_mode == 3:

        if zero_is_scalar:

            b = (b - zeros).to(meta_dtype) * scales
        else:

            b = (b.to(meta_dtype) - zeros) * scales

    if W_group_mode == 4:

        b = tl.fma(b.to(meta_dtype), scales, zeros)

    return b


@triton.jit
def swizzle_tile(
    pid,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@triton.jit
def linear_tile(
    pid,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    return pid_m, pid_n


@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    scales_a_ptr,
    M,
    N,
    K,
    W_nbits: tl.constexpr,
    group_size: tl.constexpr,
    unpack_mask: tl.constexpr,
    elements_per_sample: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_meta_g,
    stride_meta_n,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_load_order: tl.constexpr,
    meta_evict_policy: tl.constexpr,
    data_contiguous: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if data_contiguous:

        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    a_mask = offs_am[:, None] < M

    b_ptrs = b_ptr + (
        (offs_k[:, None] // elements_per_sample) * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]

    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs = zeros_ptr + offs_bn[None, :] * stride_meta_n

    stride_mul = BLOCK_SIZE_K / group_size

    if zero_is_scalar:
        zero_scalar = tl.load(zeros_ptr, eviction_policy="evict_last")

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(num_pid_k):

        if A_load_order == 0:
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")

        b = tl.load(b_ptrs, eviction_policy="evict_first")

        if A_load_order == 1:
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")

        if W_group_mode > 0:

            k_m = (k * stride_mul).to(tl.int32)

        if W_group_mode >= 2:
            scales = tl.load(
                scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy
            )
        else:
            scales = None

        if W_group_mode == 1 or W_group_mode >= 3:
            if zero_is_scalar:

                zeros = zero_scalar
            else:

                zeros = tl.load(
                    zeros_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy
                )
        else:
            zeros = None

        if A_load_order == 2:
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")

        b = dequantize(
            b,
            scales,
            zeros,
            q_shift,
            meta_dtype,
            unpack_mask,
            elements_per_sample,
            W_group_mode,
            zero_is_scalar,
        )

        if A_load_order == 3:
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")

        acc = tl.dot(
            a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype, input_precision="tf32"
        )

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // elements_per_sample) * stride_bk

    if channel_scale_mode == 1:

        scales_b = tl.load(
            scales_ptr + offs_bn,
            mask=offs_bn < N,
            other=1,
            eviction_policy=meta_evict_policy,
        )

        acc = acc.to(meta_dtype) * scales_b[None, :]

    if channel_scale_mode == 2:

        scales_a = tl.load(
            scales_a_ptr + offs_am,
            mask=offs_am < M,
            other=1,
            eviction_policy=meta_evict_policy,
        )

        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)

        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if channel_scale_mode == 3:

        scales_a = tl.load(
            scales_a_ptr + offs_am,
            mask=offs_am < M,
            other=1,
            eviction_policy=meta_evict_policy,
        )
        scales_b = tl.load(
            scales_ptr + offs_bn,
            mask=offs_bn < N,
            other=1,
            eviction_policy=meta_evict_policy,
        )

        acc = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    acc = acc.to(output_dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def test_kernel():

    M, N, K = 128, 256, 512
    W_nbits = 4
    group_size = 64
    elements_per_sample = 8

    a = torch.randn(M, K, dtype=torch.float16, device="cuda")

    b_unpacked = torch.randint(-8, 7, (K, N), dtype=torch.int8, device="cuda")

    b_packed = torch.zeros(
        (K // elements_per_sample, N), dtype=torch.int32, device="cuda"
    )
    for i in range(elements_per_sample):
        b_packed |= (
            b_unpacked[i::elements_per_sample, :].to(torch.int32) & ((1 << W_nbits) - 1)
        ) << (i * W_nbits)

    scales = torch.ones((K // group_size, N), dtype=torch.float16, device="cuda")
    zeros = torch.zeros((K // group_size, N), dtype=torch.float16, device="cuda")
    scales_a = torch.ones(M, dtype=torch.float16, device="cuda")

    c_triton = torch.zeros((M, N), dtype=torch.float16, device="cuda")

    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b_packed.stride(0), b_packed.stride(1)
    stride_cm, stride_cn = c_triton.stride(0), c_triton.stride(1)
    stride_meta_g, stride_meta_n = scales.stride(0), scales.stride(1)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    b_dequantized = torch.zeros((K, N), dtype=torch.float16, device="cuda")
    for g in range(K // group_size):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, K)
        for i in range(start_idx, end_idx):
            element_idx = i % elements_per_sample
            packed_idx = i // elements_per_sample
            shift = element_idx * W_nbits
            mask = (1 << W_nbits) - 1
            b_dequantized[i] = ((b_packed[packed_idx] >> shift) & mask).to(
                torch.float16
            )

            b_dequantized[i] *= scales[g]

    c_ref = torch.matmul(a, b_dequantized)

    max_diff = torch.max(torch.abs(c_ref - c_triton))
    print(f"Max difference between PyTorch and Triton: {max_diff.item()}")

    warmup = 25
    rep = 100

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(warmup + rep):
        gemm_kernel[grid](
            a_ptr=a,
            b_ptr=b_packed,
            c_ptr=c_triton,
            scales_ptr=scales,
            zeros_ptr=zeros,
            scales_a_ptr=scales_a,
            M=M,
            N=N,
            K=K,
            W_nbits=W_nbits,
            group_size=group_size,
            unpack_mask=(1 << W_nbits) - 1,
            elements_per_sample=elements_per_sample,
            stride_am=stride_am,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            stride_bn=stride_bn,
            stride_cm=stride_cm,
            stride_cn=stride_cn,
            stride_meta_g=stride_meta_g,
            stride_meta_n=stride_meta_n,
            input_dtype=tl.float16,
            output_dtype=tl.float16,
            acc_dtype=tl.float32,
            meta_dtype=tl.float16,
            channel_scale_mode=1,
            W_group_mode=1,
            zero_is_scalar=0,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=16,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
            A_load_order=1,
            meta_evict_policy="evict_last",
            data_contiguous=1,
        )
    torch.cuda.synchronize()
    end = time.time()

    elapsed_time = (end - start) / rep
    print(f"Triton kernel time: {elapsed_time * 1000:.2f} ms")

    return c_triton, c_ref, max_diff.item()


if __name__ == "__main__":
    test_kernel()
