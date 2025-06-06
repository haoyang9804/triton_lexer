import os
from typing import Generator, List

import torch
import triton

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernels import triton_add_kernel


class Operator(BenchmarkOperator):
    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics):
        return (
            3
            * example_inputs[0].element_size()
            * example_inputs[0].numel()
            / metrics.latency
            * 1e-6
        )

    @register_benchmark()
    def triton_add(self, x: torch.Tensor, y: torch.Tensor):

        output = torch.empty_like(x)
        n_elements = output.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        def _inner():
            return_val = triton_add_kernel[grid](
                x, y, output, n_elements, BLOCK_SIZE=1024
            )
            return return_val

        return _inner

    @register_benchmark(baseline=True)
    def torch_add(self, x: torch.Tensor, y: torch.Tensor):
        return lambda: x + y

    def get_x_vals(self) -> List[int]:
        return [2**i for i in range(12, 28, 1)]

    def get_x_val(self, example_inputs):
        return len(example_inputs[0])

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["size"],
                x_vals=self.x_vals,
                x_log=True,
                line_arg="provider",
                line_vals=[
                    "torch_add",
                    "triton_add",
                ],
                line_names=["Torch", "Triton"],
                styles=[("blue", "-"), ("green", "-")],
                ylabel="GB/s",
                plot_name="vector-add-performance",
                args={},
            )
        )
        def _plot(size, provider):
            gbps, max_gbps, min_gbps = self.output.get_y_vals(size, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/vector_add")

    def get_input_iter(self) -> Generator:
        for size in self.get_x_vals():
            x = torch.rand(size, device=self.device, dtype=self.dtype)
            y = torch.rand(size, device=self.device, dtype=self.dtype)
            yield x, y
