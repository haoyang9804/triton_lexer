import torch
import triton
import triton.language as tl
import math

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


def naive_CELoss(x, E, targets):

    logits = x @ E

    B, N, _ = x.shape
    V = E.shape[1]
    logits_2d = logits.reshape(-1, V)

    targets_1d = targets.reshape(-1)

    max_logits, _ = torch.max(logits_2d, dim=1, keepdim=True)
    logits_shifted = logits_2d - max_logits
    log_sum_exp = (
        torch.log(torch.sum(torch.exp(logits_shifted), dim=1, keepdim=True))
        + max_logits
    )
    log_softmax = logits_2d - log_sum_exp

    nll = -log_softmax[
        torch.arange(log_softmax.size(0), device=targets_1d.device), targets_1d
    ]

    loss = torch.mean(nll)

    return loss


@triton.autotune(
    [
        triton.Config(
            {"bsN": bsN, "bsD": bsD, "bsV": bsV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for bsN in [16]
        for bsD in [16]
        for bsV in [16]
        for num_stages in [3]
        for num_warps in [4]
    ],
    key=["N", "D", "V"],
)
@triton.jit
def fused_CELoss_kernel(
    x_ptr,
    E_ptr,
    targets_ptr,
    out_ptr,
    stride_x_B,
    stride_x_N,
    stride_x_D,
    stride_E_D,
    stride_E_V,
    stride_tar_B,
    stride_tar_N,
    stride_out_B,
    stride_out_N,
    B,
    N,
    D: tl.constexpr,
    V: tl.constexpr,
    bsN: tl.constexpr,
    bsD: tl.constexpr,
    bsV: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_N = tl.arange(0, bsN)
    offsets_V = tl.arange(0, bsV)

    x_ptr += pid * bsN * stride_x_B
    targets_ptr += pid * bsN * stride_tar_N
    out_ptr += pid * bsN * stride_out_N

    M = tl.full((bsN,), value=-1e6, dtype=tl.float32)
    denominator = tl.full((bsN,), value=1.0, dtype=tl.float32)
    numerator_selected = tl.zeros((bsN,), dtype=tl.float32)

    targets = tl.load(targets_ptr + offsets_N * stride_tar_N).to(tl.int32)

    for block_start_outer in range(0, V, bsV):

        logits = tl.zeros((bsN, bsV), dtype=tl.float32)
        offsets_D = tl.arange(0, bsD)

        for block_start_inner in range(0, D, bsD):

            x_offsets = (
                offsets_N[:, None] * stride_x_N + offsets_D[None, :] * stride_x_D
            )
            E_offsets = (
                offsets_D[:, None] * stride_E_D + offsets_V[None, :] * stride_E_V
            )
            x = tl.load(x_ptr + x_offsets)
            E = tl.load(E_ptr + E_offsets)
            logits = tl.dot(x, E, acc=logits)

            offsets_D += bsD
        offsets_V += bsV

        M_new = tl.maximum(M, tl.max(logits, axis=1))

        logits_shifted = logits - M_new[:, None]
        numerator = tl.exp(logits_shifted)
        alpha = tl.exp(M - M_new)
        denominator_new = tl.sum(numerator, axis=1)
        denominator = denominator * alpha + denominator_new

        targets_adj = targets - block_start_outer

        mask = tl.arange(0, bsV)[None, :] == targets_adj[:, None]
        numerator_selected += tl.sum(tl.where(mask, numerator, 0.0), axis=1)

        M = M_new

    P = numerator_selected / denominator
    nll = -tl.log(P)

    tl.store(out_ptr + offsets_N * stride_out_N, nll)


def fused_CELoss(x, E, targets):
    assert x.shape[-1] == E.shape[0]
    B, N, D = x.shape
    _, V = E.shape

    out = torch.empty((B, N), dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(B * N, meta["bsN"]),)

    fused_CELoss_kernel[grid](
        x,
        E,
        targets,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        E.stride(0),
        E.stride(1),
        targets.stride(0),
        targets.stride(1),
        out.stride(0),
        out.stride(1),
        B,
        N,
        D,
        V,
    )

    return torch.mean(out)


def test_naiveCELoss(B, N, D, V, device=DEVICE, atol=1e-3):
    torch.cuda.empty_cache()
    assert V <= 32_768

    x = torch.randn((B, N, D), dtype=torch.float32, device=device, requires_grad=False)
    E = torch.randn((D, V), dtype=torch.float32, device=device, requires_grad=False)
    targets = torch.randint(0, V, (B, N), device=device, requires_grad=False)

    naive_loss = naive_CELoss(x, E, targets)
    logits = (x @ E).reshape(-1, V)
    targets_1d = targets.reshape(-1)
    ref_loss = torch.nn.functional.cross_entropy(logits, targets_1d)

    torch.testing.assert_close(naive_loss, ref_loss, atol=atol, rtol=0)
    print(f"naive passed {V}")


def test_fusedCELoss(B, N, D, V, device=DEVICE, atol=1e-3):
    torch.cuda.empty_cache()

    x = torch.randn((B, N, D), dtype=torch.float32, device=device, requires_grad=False)
    E = torch.randn((D, V), dtype=torch.float32, device=device, requires_grad=False)
    targets = torch.randint(0, V, (B, N), device=device, requires_grad=False)

    logits = (x @ E).reshape(-1, V)
    targets_1d = targets.reshape(-1)
    ref_loss = torch.nn.functional.cross_entropy(logits, targets_1d)
    tri_loss = fused_CELoss(x, E, targets)

    torch.testing.assert_close(tri_loss, ref_loss, atol=atol, rtol=0)
    print(f"triton passed {V}")


configs = [
    triton.testing.Benchmark(
        x_names=["V"],
        x_vals=[2**i for i in range(10, 14)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=[
            "torch.nn.functional.cross_entropy()",
            "Fused & sparse Triton implementation",
        ],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name=f"CELoss-performance",
        args={},
    )
]


@triton.testing.perf_report(configs)
def bench_CELoss(V, provider, device=DEVICE):
    dtype = torch.float32
    B, N, D = 32, 1024, 384
    x = torch.randn((B, N, D), dtype=dtype, device=device, requires_grad=False)
    E = torch.randn((D, V), dtype=dtype, device=device, requires_grad=False)
    targets = torch.randint(0, V, (B, N), device=device, requires_grad=False)
    if provider == "torch":
        logits = (x @ E).reshape(-1, V)
        targets_1d = targets.reshape(-1)
        fn = lambda: torch.nn.functional.cross_entropy(logits, targets_1d)
    if provider == "triton":
        fn = lambda: fused_CELoss(x, E, targets)

    ms = triton.testing.do_bench(fn)

    total_flops = 2 * B * N * D * V + 6 * B * N * V
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":

    test_fusedCELoss(32, 1024, 384, 32_768)

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        bench_CELoss.run(save_path=".", print_data=True)
