from typing import Generator, List

import torch
import triton
import triton.language as tl

from tritonbench.utils.data_utils import get_production_shapes

from tritonbench.utils.env_utils import is_fbcode

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)


class Operator(BenchmarkOperator):
    is_compute_bound = False

    @register_benchmark()
    def triton_softmax(self, x):
        n_rows, n_cols = x.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16

        y = torch.empty_like(x)

        def _inner():
            Operator.softmax_kernel[(n_rows,)](
                y,
                x,
                x.stride(0),
                y.stride(0),
                n_cols,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return y

        return _inner

    @triton.jit
    def softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):

        row_idx = tl.program_id(0)

        row_start_ptr = input_ptr + row_idx * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))

        row_minus_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    @register_benchmark(baseline=True)
    def naive_softmax(self, x):

        def _inner():

            x_max = x.max(dim=1)[0]

            z = x - x_max[:, None]

            numerator = torch.exp(z)

            denominator = numerator.sum(dim=1)

            ret = numerator / denominator[:, None]

            return ret

        return _inner

    def get_input_iter(self):
        M = 4096
        shapes = [(M, 128 * i) for i in range(2, 100)]
        if is_fbcode() and self.tb_args.production_shapes:
            additional_shapes = get_production_shapes(
                self.name, "softmax", self.tb_args.shuffle_shapes
            )
            if additional_shapes:
                shapes.extend(additional_shapes)
        for M, N in shapes:
            yield (torch.randn([M, N], dtype=self.dtype, device=self.device),)

    def get_x_val(self, example_inputs):
        shape = example_inputs[0].size()
        return [shape[0], shape[1]]

    @register_metric()
    def gbps(self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics) -> float:
        return (
            2
            * example_inputs[0].nelement()
            * example_inputs[0].element_size()
            * 1e-9
            / (metrics.latency * 1e-3)
        )

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=[
                    "triton_softmax",
                    "naive_softmax",
                ],
                line_names=[
                    "Triton",
                    "Torch (native)",
                ],
                styles=[("blue", "-"), ("green", "-"), ("green", "--")],
                ylabel="GB/s",
                plot_name="softmax-performance",
                args={"M": 4096},
            )
        )
        def _plot(M, N, provider):
            gbps, max_gbps, min_gbps = self.output.get_y_vals(N, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_softmax")
