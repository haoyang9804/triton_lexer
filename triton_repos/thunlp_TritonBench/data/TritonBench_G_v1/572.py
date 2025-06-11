import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["D"],
)
@triton.heuristics({"HAS_SCALE": lambda args: args["scale"] is not None})
@triton.jit
def logsumexp_fwd_kernel(
    x, z, scale, D: tl.constexpr, B: tl.constexpr, HAS_SCALE: tl.constexpr
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float("inf"))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)


def logsumexp_fwd(
    x, scale: Optional[float] = None, dtype: Optional[torch.dtype] = None
):
    r

    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[(N, ND)](x=x, z=z, scale=scale, D=D, B=B)
    z = z.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z


def test_logsumexp_fwd():
    batch_size = 4
    seq_len = 64
    scale = 0.5

    x = torch.randn((batch_size, seq_len), device="cuda", dtype=torch.float32)
    z1 = logsumexp_fwd(x)

    x = torch.randn((batch_size, seq_len), device="cuda", dtype=torch.float32)
    z2 = logsumexp_fwd(x, scale=scale)

    x = torch.randn((batch_size, 16, seq_len), device="cuda", dtype=torch.float32)
    z3 = logsumexp_fwd(x)

    x = torch.randn((batch_size, seq_len), device="cuda", dtype=torch.float32)
    z4 = logsumexp_fwd(x, dtype=torch.float64)

    results = {
        "test_case_1": z1,
        "test_case_2": z2,
        "test_case_3": z3,
        "test_case_4": z4,
    }
    return results


result_gold = test_logsumexp_fwd()
