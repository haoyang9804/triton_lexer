import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TritonBench_v1.nested_loops_processing import wrapper_nested3
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__(
            "nested_loops_processing", dtype=dtype, is_backward=is_backward, **kwargs
        )

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(12, 24):
            size = 2**i
            n_rows = size
            n_cols = 4
            input_tensor = (n_rows, n_cols)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):

        return input_tensor

    def call_op(self, input_tensor):
        n_rows, n_cols = input_tensor
        wrapper_nested3(n_rows, n_cols)

    def get_gbps(self, input_tensor, runtime):
        n_rows, n_cols = input_tensor
        total_bytes = 2 * n_rows * n_cols * 4
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        n_rows, n_cols = input_tensor

        FLOPS = 3 * n_rows * n_cols
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == "__main__":
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
