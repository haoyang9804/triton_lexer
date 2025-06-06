import torch
import triton
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from fla.ops.simple_gla import chunk_simple_gla


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[64] + [128 * 2**i for i in range(0, 8)],
        line_arg="provider",
        line_vals=["chunk_simple_gla", "mamba2_ssd"],
        line_names=["chunk_simple_gla", "mamba2_ssd"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="Execution Time (ms)",
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):

    from fla.utils import device

    dtype = torch.bfloat16
    B, H, D = 16, 8, 128

    final_state = False

    X_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)
    dt_mamba = torch.ones(B, T, H, dtype=dtype, device=device)
    A_mamba = -0.1 * torch.rand(H, dtype=dtype, device=device)
    B_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)
    C_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "chunk_simple_gla":

        q = C_mamba.transpose(1, 2).contiguous()
        k = B_mamba.transpose(1, 2).contiguous()
        v = X_mamba.transpose(1, 2).contiguous()
        g = (A_mamba * dt_mamba).transpose(1, 2).contiguous()

        results = triton.testing.do_bench(
            lambda: chunk_simple_gla(
                q, k, v, g, scale=1.0, output_final_state=final_state
            ),
            quantiles=quantiles,
        )

    elif provider == "mamba2_ssd":

        results = triton.testing.do_bench(
            lambda: mamba_chunk_scan_combined(
                X_mamba,
                dt_mamba,
                A_mamba,
                B_mamba,
                C_mamba,
                chunk_size=64,
                D=None,
                return_final_states=final_state,
            ),
            quantiles=quantiles,
        )
    return results


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".")
