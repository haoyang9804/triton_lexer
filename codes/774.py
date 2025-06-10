import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TritonBench_v1.fast_rope_embedding import fast_rope_embedding
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl


class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__(
            "fast_rope_embedding", dtype=dtype, is_backward=is_backward, **kwargs
        )

    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 16):
            batch_size = 2**i
            seq_len = 128
            n_heads = 8
            head_dim = 64
            Q = torch.rand(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32)
            K = torch.rand(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32)
            cos = torch.rand(seq_len, head_dim // 2, dtype=torch.float32)
            sin = torch.rand(seq_len, head_dim // 2, dtype=torch.float32)
            self.input_tensors.append((Q, K, cos, sin))

    def to_cuda(self, input_tensor):
        Q, K, cos, sin = input_tensor
        return (Q.cuda(), K.cuda(), cos.cuda(), sin.cuda())

    def call_op(self, input_tensor):
        Q, K, cos, sin = input_tensor
        return fast_rope_embedding(Q, K, cos, sin)

    def get_gbps(self, input_tensor, runtime):
        Q, _, _, _ = input_tensor
        total_bytes = 2 * Q.numel() * Q.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS

    def get_tflops(self, input_tensor, runtime):
        Q, _, _, _ = input_tensor
        FLOPS = 4 * Q.numel()
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS


if __name__ == "__main__":
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
