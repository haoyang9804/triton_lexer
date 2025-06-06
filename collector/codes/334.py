import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[128 * 2**i for i in range(0, 8)],
        line_arg="provider",
        line_vals=[
            "naive",
            "fused",
            "fused_linear",
            "naive_bwd",
            "fused_bwd",
            "fused_linear_bwd",
        ],
        line_names=[
            "naive",
            "fused",
            "fused_linear",
            "naive_bwd",
            "fused_bwd",
            "fused_linear_bwd",
        ],
        styles=[
            ("green", "-"),
            ("blue", "--"),
            ("red", "-."),
            ("cyan", ":"),
            ("yellow", "dotted"),
            ("cyan", "--"),
            ("cyan", "-"),
            ("black", ":"),
        ],
        ylabel="Execution Time (ms)",
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    from fla.utils import device

    dtype = torch.bfloat16
    requires_grad = True
    B, H, V = 4, 4096, 120000

    x = torch.randn(B * T, H, device=device, requires_grad=requires_grad, dtype=dtype)
    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.int64)
    w = torch.randn(V, H, device=device, requires_grad=requires_grad, dtype=dtype)
    b = torch.randn(V, device=device, requires_grad=requires_grad, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == "naive":
        criterion = nn.CrossEntropyLoss()
        results = triton.testing.do_bench(
            lambda: criterion(F.linear(x, w, b), target), quantiles=quantiles
        )
    elif provider == "naive_bwd":
        criterion = nn.CrossEntropyLoss()
        results = triton.testing.do_bench(
            lambda: criterion(F.linear(x, w, b), target).backward(), quantiles=quantiles
        )
    elif provider == "fused":
        criterion = FusedCrossEntropyLoss()
        results = triton.testing.do_bench(
            lambda: criterion(F.linear(x, w, b), target), quantiles=quantiles
        )
    elif provider == "fused_bwd":
        criterion = FusedCrossEntropyLoss()
        results = triton.testing.do_bench(
            lambda: criterion(F.linear(x, w, b), target).backward(), quantiles=quantiles
        )
    elif provider == "fused_linear":
        criterion = FusedLinearCrossEntropyLoss()
        results = triton.testing.do_bench(
            lambda: criterion(x, target, w, b), quantiles=quantiles
        )
    elif provider == "fused_linear_bwd":
        criterion = FusedLinearCrossEntropyLoss()
        results = triton.testing.do_bench(
            lambda: criterion(x, target, w, b).backward(), quantiles=quantiles
        )
    return results


if __name__ == "__main__":
    benchmark.run(print_data=True)
