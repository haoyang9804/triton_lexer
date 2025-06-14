import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({"BT": 16}, num_warps=2),
        triton.Config({"BT": 32}, num_warps=4),
        triton.Config({"BT": 32}, num_warps=2),
        triton.Config({"BT": 64}, num_warps=8),
        triton.Config({"BT": 64}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s,
    o,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_global_reversed_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    dtype = dtype or s.dtype
    grid = (B * H,)
    z = torch.empty_like(s, dtype=dtype)
    chunk_global_reversed_cumsum_scalar_kernel[grid](s, z, T=T)
    return z


import torch


def test_chunk_global_reversed_cumsum_scalar():
    B, H, T = 2, 3, 4
    results = {}

    s1 = torch.rand((B, H, T), dtype=torch.float32).cuda()
    result1 = chunk_global_reversed_cumsum_scalar(s1)
    results["test_case_1"] = result1

    s2 = torch.rand((B, H, T), dtype=torch.float32).cuda()
    result2 = chunk_global_reversed_cumsum_scalar(s2)
    results["test_case_2"] = result2

    s3 = torch.rand((B, H, T), dtype=torch.float32).cuda()
    result3 = chunk_global_reversed_cumsum_scalar(s3)
    results["test_case_3"] = result3

    s4 = torch.rand((B, H, T), dtype=torch.float32).cuda()
    result4 = chunk_global_reversed_cumsum_scalar(s4)
    results["test_case_4"] = result4

    return results


result_gold = test_chunk_global_reversed_cumsum_scalar()
