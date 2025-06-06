import argparse
import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import fbgemm_gpu.experimental.gen_ai

import torch
import triton

from tritonbench.utils.env_utils import is_cuda

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    gemm_shapes,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchBench FBGEMM operator Benchmark")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--llama", action="store_true")
    args = parser.parse_args(args)
    return args


HAS_TRITON = False
if is_cuda():
    try:
        from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
            matmul_fp8_block as triton_fp8_block,
        )

        HAS_TRITON = True
    except:
        HAS_TRITON = False


HAS_CUTLASS = False
if is_cuda():
    try:
        cutlass_fp8_block = torch.ops.llama_cpp.fp8_blockwise_matmul
        HAS_CUTLASS = True
    except:
        try:
            import fbgemm_gpu.experimental.gen_ai

            cutlass_fp8_block = torch.ops.fbgemm.f8f8bf16_blockwise
            HAS_CUTLASS = True
        except:
            HAS_CUTLASS = False


BUILDIN_SHAPES = [
    (1, 2304, 2048),
    (1, 8192, 16384),
    (4, 4096, 2304),
    (4, 13312, 2048),
    (8, 2304, 2304),
    (8, 8192, 6656),
    (16, 4096, 6656),
    (16, 13312, 13312),
    (32, 2304, 16384),
    (32, 8192, 13312),
    (64, 4096, 2048),
    (64, 13312, 2048),
    (128, 2304, 6656),
    (128, 8192, 2304),
    (2048, 8192, 2048),
    (2048, 13312, 6656),
    (4096, 2304, 13312),
    (4096, 13312, 2304),
    (16384, 4096, 16384),
    (16384, 8192, 13312),
]

E4M3_MAX_POS: float = torch.finfo(torch.float8_e4m3fn).max
EPS: float = 1e-12
FP16_MAX_POS: float = torch.finfo(torch.float16).max


def fp8_block_quantize(
    x: torch.Tensor, block_m: int = 128, block_k: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)

    padded_m = grid_m * block_m
    padded_k = grid_k * block_k
    x_padded = torch.zeros(padded_m, padded_k, dtype=x.dtype, device=x.device)
    x_padded[:M, :K] = x

    block_max = (
        x_padded.abs().reshape(grid_m, block_m, grid_k, block_k).amax(dim=(1, 3))
    )

    block_max = torch.clamp(block_max, min=EPS)
    x_scale = torch.empty((grid_m, grid_k), dtype=torch.float32)
    x_scale = E4M3_MAX_POS / block_max.to(torch.float32)

    x_scale[x_scale == float("inf")] = 1.0
    x_fp8 = (
        x_padded
        * x_scale.repeat_interleave(block_m, dim=0).repeat_interleave(block_k, dim=1)
    )[:M, :K]

    x_fp8 = x_fp8.to(torch.float8_e4m3fn)
    del x, x_padded
    return x_fp8, 1 / x_scale


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "speedup", "accuracy"]
    DEFAULT_PRECISION = "fp8"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        addmm_args = parse_args(self.extra_args)
        if addmm_args.m and addmm_args.n and addmm_args.k:
            self.shapes = [(addmm_args.m, addmm_args.n, addmm_args.k)]
        elif addmm_args.llama:
            self.shapes = gemm_shapes()
        else:
            self.shapes = BUILDIN_SHAPES

    @register_benchmark(enabled=HAS_TRITON)
    def _triton(self, xq, wq, x_scale, w_scale) -> Callable:
        return lambda: triton_fp8_block(xq, wq, x_scale, w_scale)

    @register_benchmark(enabled=HAS_CUTLASS, baseline=True)
    def _cutlass(self, xq, wq, x_scale, w_scale) -> Callable:
        return lambda: cutlass_fp8_block(xq, wq, x_scale, w_scale)

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> List[float]:
        xq, wq, _, _ = example_inputs
        m, k = xq.size()
        n, k = wq.size()
        flops = m * k * 2 * n
        return flops

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        xq, wq, _, _ = example_inputs
        m, k = xq.size()
        n, k = wq.size()
        return (m, n, k)

    def get_input_iter(self) -> Generator:
        for shape in self.shapes:
            m, n, k = shape
            x = torch.randn(
                (m, k), device=self.device, dtype=torch.bfloat16
            ).requires_grad_(False)
            w = torch.randn(
                (n, k), device=self.device, dtype=torch.bfloat16
            ).requires_grad_(False)
            xq, x_scale = fp8_block_quantize(x)
            wq, w_scale = fp8_block_quantize(w)
            yield xq, wq, x_scale, w_scale

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        accuracy = True
        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-2, rtol=0.5)
        except Exception:
            accuracy = False
        finally:
            return accuracy

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["density"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=[
                    "_torch",
                    "_triton",
                    "_cutlass",
                ],
                line_names=[
                    "Torch",
                    "Triton",
                    "Cutlass",
                ],
                styles=[("blue", "-"), ("green", "-"), ("yellow", "-")],
                ylabel="tflops",
                plot_name="gemm-performance",
                args={},
            )
        )
        def _plot(density, provider):
            tflops = self.output.get_y_vals(density, provider, "tflops")
            return tflops

        save_path = "/tmp/test_fp8_gemm_blockwise"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
