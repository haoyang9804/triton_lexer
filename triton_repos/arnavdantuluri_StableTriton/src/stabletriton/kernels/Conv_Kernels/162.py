import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "TILE_SIZE_M": 64,
                "TILE_SIZE_N": 32,
                "TILE_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            },
            num_stages=5,
            num_warps=4,
        ),
    ],
    key=["N", "C", "H", "W", "R", "S", "K"],
)
@triton.jit
def implicit_gemm_fprop_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    C,
    H,
    W,
    R,
    S,
    K,
    stride_An,
    stride_Ah,
    stride_Aw,
    stride_Ac,
    stride_Bk,
    stride_Br,
    stride_Bs,
    stride_Bc,
    stride_Cn,
    stride_Cp,
    stride_Cq,
    stride_Ck,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    TILE_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    Pad_H = 1
    Pad_W = 1
    Stride_H = 1
    Stride_W = 1
    Dilation_H = 1
    Dilation_W = 1

    P = ((H + Pad_H * 2 - R * Dilation_H) // Stride_H) + 1
    Q = ((W + Pad_W * 2 - S * Dilation_W) // Stride_W) + 1
    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, TILE_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, TILE_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pq = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    q = pq % Q
    p = pq // Q

    k = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)

    crs = tl.arange(0, TILE_SIZE_K)
    s = crs % S
    c = (crs // S) // R
    r = (crs // S) % R

    a_ptrs = (
        a_ptr
        + q[:, None] * stride_Aw
        + p[:, None] * stride_Ah
        + r[None, :] * stride_Ah
        + s[None, :] * stride_Aw
        + c[None, :] * stride_Ac
    )

    b_ptrs = (
        b_ptr
        + r[:, None] * stride_Br
        + s[:, None] * stride_Bs
        + c[:, None] * stride_Bc
        + k[None, :] * stride_Bk
    )

    accumulator = tl.zeros((TILE_SIZE_M, TILE_SIZE_N), dtype=tl.float32)
    for gemm_k in range(0, GEMM_K, TILE_SIZE_K):

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        accumulator += tl.dot(a, b)

        crs += TILE_SIZE_K
        s = crs % S
        c = (crs // S) // R
        r = (crs // S) % R

        a_ptrs = (
            a_ptr
            + q[:, None] * stride_Aw
            + p[:, None] * stride_Ah
            + r[None, :] * stride_Ah
            + s[None, :] * stride_Aw
            + c[None, :] * stride_Ac
        )

        b_ptrs = (
            b_ptr
            + r[:, None] * stride_Br
            + s[:, None] * stride_Bs
            + c[:, None] * stride_Bc
            + k[None, :] * stride_Bk
        )

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    offs_cn = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)

    c_ptrs = c_ptr + stride_Cq * offs_cm[:, None] + stride_Ck * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < GEMM_M) & (offs_cn[None, :] < GEMM_N)
    tl.store(c_ptrs, c, mask=c_mask)


def implicit_gemm_fprop(a, b, activation=None):

    assert a.shape[3] == b.shape[3], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    N, H, W, C = a.shape
    K, R, S, C = b.shape
    assert (
        C * R * S
    ) % 32 == 0, "We don't check memory-out-of-bounds with GEMM_K so GEMM_K must be divisible by BLOCK_SIZE_K"

    Pad_H = 1
    Pad_W = 1
    Stride_H = 1
    Stride_W = 1
    Dilation_H = 1
    Dilation_W = 1
    P = ((H + Pad_H * 2 - R * Dilation_H) // Stride_H) + 1
    Q = ((W + Pad_W * 2 - S * Dilation_W) // Stride_W) + 1

    c = torch.empty((N, P, Q, K), device=a.device, dtype=a.dtype)

    GEMM_M = N * P * Q
    GEMM_N = K
    grid = lambda META: (
        triton.cdiv(GEMM_M, META["TILE_SIZE_M"])
        * triton.cdiv(GEMM_N, META["TILE_SIZE_N"]),
    )
    implicit_gemm_fprop_kernel[grid](
        a,
        b,
        c,
        N,
        C,
        H,
        W,
        R,
        S,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a.stride(3),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b.stride(3),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        c.stride(3),
    )
    return c


torch.manual_seed(0)
UT_N = 16
UT_H = 28
UT_W = 28
UT_C = 128
UT_R = 3
UT_S = 3
UT_K = 128
a = torch.randn((UT_N, UT_H, UT_W, UT_C), device="cuda", dtype=torch.float16)
b = torch.randn((UT_K, UT_R, UT_S, UT_C), device="cuda", dtype=torch.float16)
triton_output = implicit_gemm_fprop(a, b)
conv = torch.nn.Conv2d(UT_C, UT_K, (UT_R, UT_S))
conv.weight.data = b
torch_output = conv(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if triton.testing.allclose(triton_output, torch_output):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "N",
        ],
        x_vals=[
            16,
        ],
        line_arg="provider",
        line_vals=[
            "triton",
        ],
        line_names=[
            "Triton",
        ],
        styles=[
            ("green", "-"),
        ],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={"H": 28, "W": 28, "C": 128, "K": 128},
    )
)
def benchmark(N, H, W, C, K, provider):
    a = torch.randn((N, H, W, C), device="cuda", dtype=torch.float16)
    b = torch.randn((K, 3, 3, C), device="cuda", dtype=torch.float16)
    conv = torch.nn.Conv2d(C, K, (3, 3), bias=False)
    a_torch = torch.randn((N, C, H, W), device="cuda", dtype=torch.float16)
    conv.weight.data = b
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: conv(a_torch))
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: implicit_gemm_fprop(a, b))
    print(
        "N: {}, H: {}, W: {}, C: {}, K: {} latency: {}/{}/{}".format(
            N, H, W, C, K, ms, min_ms, max_ms
        )
    )
    perf = lambda ms: 2 * N * H * W * K * C * 3 * 3 * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
