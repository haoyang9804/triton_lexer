import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def rms_norm_fwd_fused(
    X,
    Y,
    W,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):

    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w

        tl.store(Y + cols, y, mask=mask)


class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):

        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        y = torch.empty_like(x)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        rms_norm_fwd_fused[(M,)](
            x_arg,
            y,
            self.weight,
            x_arg.stride(0),
            N,
            self.variance_epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y


def test_triton_llama_rms_norm():
    results = {}

    x1 = torch.randn(2, 16, dtype=torch.float32, device="cuda")
    weight1 = torch.ones(16, dtype=torch.float32, device="cuda")
    norm1 = TritonLlamaRMSNorm(weight1)
    y1 = norm1(x1)
    results["test_case_1"] = y1

    x2 = torch.randn(4, 256, dtype=torch.float32, device="cuda")
    weight2 = torch.ones(256, dtype=torch.float32, device="cuda")
    norm2 = TritonLlamaRMSNorm(weight2)
    y2 = norm2(x2)
    results["test_case_2"] = y2

    x3 = torch.randn(1, 65536 // 4, dtype=torch.float32, device="cuda")
    weight3 = torch.ones(65536 // 4, dtype=torch.float32, device="cuda")
    norm3 = TritonLlamaRMSNorm(weight3)
    y3 = norm3(x3)
    results["test_case_3"] = y3

    try:
        x4 = torch.randn(1, 65536 // 4 + 1, dtype=torch.float32, device="cuda")
        weight4 = torch.ones(65536 // 4 + 1, dtype=torch.float32, device="cuda")
        norm4 = TritonLlamaRMSNorm(weight4)
        y4 = norm4(x4)
    except RuntimeError as e:
        results["test_case_4"] = str(e)

    return results


result_gold = test_triton_llama_rms_norm()
