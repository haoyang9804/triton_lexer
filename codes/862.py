import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TritonBench_v1.rope_backward_transform import rope_backward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__(
            "rope_backward_transform", dtype=dtype, is_backward=is_backward, **kwargs
        )

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):
            batch_size = 2**i
            seq_len = 128
            n_q_head = 8
            n_kv_head = 8
            head_dim = 64

            dq = torch.rand(
                batch_size, n_q_head, seq_len, head_dim, dtype=torch.float16
            )
            dk = torch.rand(
                batch_size, n_kv_head, seq_len, head_dim, dtype=torch.float16
            )
            cos = torch.rand(seq_len, head_dim // 2, dtype=torch.float16)
            sin = torch.rand(seq_len, head_dim // 2, dtype=torch.float16)

            self.input_tensors.append((dq, dk, cos, sin))

    def to_cuda(self, input_tensor):
        dq, dk, cos, sin = input_tensor
        return dq.cuda(), dk.cuda(), cos.cuda(), sin.cuda()

    def call_op(self, input_tensor):
        dq, dk, cos, sin = input_tensor
        return rope_backward(dq, dk, cos, sin)

    def get_gbps(self, input_tensor, runtime):
        dq, dk, cos, sin = input_tensor
        total_bytes = (
            dq.numel() + dk.numel() + cos.numel() + sin.numel()
        ) * dq.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        dq, dk, cos, sin = input_tensor

        FLOPS = 2 * (dq.numel() + dk.numel())
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == "__main__":
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
