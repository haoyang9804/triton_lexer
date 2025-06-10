import argparse
import os
import random
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import triton

from tritonbench.utils.data_utils import get_production_shapes

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    gemm_shapes,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_args(args: List[str]) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="TorchBench fp8 gemm grouped operator Benchmark"
    )

    parser.add_argument("--m", type=int, help="The number of rows in the input matrix.")
    parser.add_argument(
        "--n", type=int, help="The number of columns in the input matrix."
    )
    parser.add_argument(
        "--k", type=int, help="The number of columns in the weight matrix."
    )
    parser.add_argument(
        "--group_size",
        type=int,
        help="The size of the groups in the grouped GEMM operation.",
    )
    parser.add_argument("--llama", action="store_true", help="Use LLaMA model shapes.")
    parser.add_argument(
        "--prefill", default=False, action="store_true", help="Use prefill shapes."
    )
    parser.add_argument(
        "--no_fp8_fast_accum",
        dest="fp8_fast_accum",
        action="store_false",
        help="Disable fast accumulation for FP8.",
    )
    parser.add_argument(
        "--no_use_tma",
        dest="use_tma",
        default=None,
        action="store_false",
        help="Disable the use of TMA (Tensor Memory Accelerator).",
    )
    parser.add_argument(
        "--use_tma",
        dest="use_tma",
        action="store_true",
        help="Enable the use of TMA (Tensor Memory Accelerator).",
    )
    parser.add_argument(
        "--no_use_persistent",
        dest="no_use_persistent",
        action="store_true",
        help="Disable the use of persistent memory.",
    )
    parser.add_argument(
        "--warp_specialization",
        action="store_true",
        help="Enable warp specialization.",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.use_tma is None:

        parsed_args.use_tma = True if torch.version.hip is None else False

    if torch.version.hip is not None:
        if parsed_args.use_tma:
            raise RuntimeError("TMA is not supported on ROCm")
        parsed_args.no_use_persistent = True
        if parsed_args.warp_specialization:
            parsed_args.warp_specialization = False
            print("Warp specialization is not supported on ROCm defaulting to False")

    return parsed_args


HAS_CUBLAS = False
HAS_TRITON = False
HAS_CUTLASS_OR_CK = False


try:
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        get_fp8_constants as get_fp8_constants,
    )
except (ImportError, AssertionError):

    HAS_TRITON = False


try:
    from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
        grouped_gemm as grouped_gemm,
        grouped_gemm_fp8_rowwise as grouped_gemm_fp8_rowwise,
    )

    HAS_TRITON = True
except (ImportError, AssertionError):

    HAS_TRITON = False


try:
    import fbgemm_gpu.experimental.gen_ai

    cutlass_or_ck_fp8_grouped_mm = torch.ops.fbgemm.f8f8bf16_rowwise_grouped_stacked

    HAS_CUTLASS_OR_CK = True
except (ImportError, AttributeError):

    HAS_CUTLASS_OR_CK = False


BUILTIN_SHAPES = [
    (1024, 1024, 1024),
    (2048, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 4096, 4096),
]

GROUP_SIZES = [
    2,
    4,
]

FP8_DTYPE, _, _, _ = get_fp8_constants()
E4M3_MAX_POS: float = torch.finfo(FP8_DTYPE).max
EPS: float = 1e-12
FP16_MAX_POS: float = torch.finfo(torch.float16).max


def fp8_row_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    row_max = torch.max(torch.abs(x), dim=1).values

    scale = E4M3_MAX_POS / torch.clamp(row_max, EPS)

    if x.dtype is torch.float16:
        scale = torch.clamp(scale, max=FP16_MAX_POS)

    xq = torch.clamp(x * scale[:, None], min=-E4M3_MAX_POS, max=E4M3_MAX_POS)

    xq = xq.to(FP8_DTYPE)

    return xq, scale.reciprocal().to(torch.float32)


def cumulative_sum_with_initial_offset(tensor):

    cumsum = torch.zeros_like(tensor)
    cumsum[1:] = torch.cumsum(tensor[:-1], dim=0)
    return cumsum


def reshape_tensor(input_tensor, m_sizes):

    G = len(m_sizes)

    N = input_tensor.size(0) // G

    K = input_tensor.size(1)

    reshaped_tensor = input_tensor.view(G, N, K)
    return reshaped_tensor


class Operator(BenchmarkOperator):

    DEFAULT_METRICS = ["tflops", "gbps", "speedup", "accuracy"]
    DEFAULT_PRECISION = "fp8"

    def __init__(
        self,
        tb_args: argparse.Namespace,
        extra_args: Optional[List[str]] = None,
    ) -> None:

        super().__init__(tb_args, extra_args)

        self.use_cuda_graphs = True

        self.fp8_fast_accum = True

        addmm_args = parse_args(self.extra_args)

        if addmm_args.m and addmm_args.n and addmm_args.k:

            self.shapes = [(addmm_args.m, addmm_args.n, addmm_args.k)]
        elif addmm_args.llama:

            self.shapes = gemm_shapes(addmm_args.prefill)
        else:

            self.shapes = BUILTIN_SHAPES

        if addmm_args.group_size:

            self.group_sizes = [addmm_args.group_size]
        else:

            self.group_sizes = GROUP_SIZES

        self.fp8_fast_accum = addmm_args.fp8_fast_accum
        self.use_tma = addmm_args.use_tma
        self.no_use_persistent = addmm_args.no_use_persistent
        self.warp_specialization = addmm_args.warp_specialization

    @register_benchmark(enabled=HAS_TRITON)
    def _triton(self, group_A, group_B, m_sizes, a_scale, b_scale) -> Callable:

        return lambda: grouped_gemm_fp8_rowwise(
            group_A,
            group_B,
            m_sizes,
            a_scale,
            b_scale,
            use_fast_accum=self.fp8_fast_accum,
            _use_warp_specialization=self.warp_specialization,
        )

    @register_benchmark(
        enabled=HAS_CUTLASS_OR_CK,
        label="ck" if torch.version.hip else "cutlass",
        baseline=True,
    )
    def _cutlass_or_ck(self, group_A, group_B, m_sizes, a_scale, b_scale) -> Callable:

        reshaped_group_B = reshape_tensor(group_B, m_sizes)

        return lambda: cutlass_or_ck_fp8_grouped_mm(
            group_A,
            reshaped_group_B,
            a_scale,
            b_scale,
            m_sizes.to(torch.int64),
        )

    @register_x_val(label="(group_size, M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int, int]:

        group_A, group_B, m_sizes, a_scale, b_scale = example_inputs

        group_size = len(m_sizes)

        xq, wq = group_A, group_B
        m, k = xq.size()
        gn, k = wq.size()

        n = gn // group_size

        return (group_size, m, n, k)

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:

        group_A, group_B, m_sizes, a_scale, b_scale = example_inputs

        xq, wq = group_A, group_B
        m, k = xq.size()
        gn, k = wq.size()
        group_size = len(m_sizes)
        n = gn // group_size

        flops = n * m * k * 2
        return flops

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:

        def nbytes(t):

            return t.numel() * t.element_size()

        group_A, group_B, m_sizes, a_scale, b_scale = example_inputs

        c = fn()

        c = c[0] if isinstance(c, tuple) else c

        gb = (nbytes(group_A) + nbytes(group_B) + nbytes(c)) / 1e9

        gbps = gb / metrics.latency * 1e3
        return gbps

    def get_input_iter(self) -> Generator:

        for group_size in self.group_sizes:
            for shape in self.shapes:

                m, n, k = shape

                B = torch.randn(
                    (group_size * n, k), device=self.device, dtype=torch.bfloat16
                ).requires_grad_(False)

                group_B, b_scale = fp8_row_quantize(B)

                m_sizes = [m // group_size] * group_size

                m_sizes = torch.tensor(m_sizes, device=self.device, dtype=torch.int32)

                A = torch.randn(
                    (m, k), device=self.device, dtype=torch.bfloat16
                ).requires_grad_(False)

                group_A, a_scale = fp8_row_quantize(A)

                yield group_A, group_B, m_sizes, a_scale, b_scale

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:

        output = fn()

        baseline_output = baseline_fn()
        try:

            torch.testing.assert_close(output, baseline_output, atol=1e-2, rtol=0.5)

            return True
        except Exception:

            return False

    def plot(self):

        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["density"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=[
                    "_triton",
                    "_ck" if torch.version.hip else "_cutlass",
                ],
                line_names=[
                    "Triton",
                    "CK" if torch.version.hip else "Cutlass",
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

        save_path = "/tmp/test_fp8_gemm_grouped"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
