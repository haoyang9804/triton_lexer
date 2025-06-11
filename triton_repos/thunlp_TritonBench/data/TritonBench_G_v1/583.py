import torch
import triton
import triton.language as tl


@triton.jit
def matmul_tma_load_store(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_F16: tl.constexpr,
):

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    c = tl.dot(a, b)

    if OUTPUT_F16:
        c = c.to(tl.float16)

    tl.store(c_block_ptr, c)


def warpper_tma_load_store(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_F16):

    if TRANS_A:
        a = torch.randn((K, M), device="cuda", dtype=torch.float16).T
    else:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    if TRANS_B:
        b = torch.randn((N, K), device="cuda", dtype=torch.float16).T
    else:
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    if OUTPUT_F16:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    matmul_tma_load_store[(1, 1)](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        OUTPUT_F16=OUTPUT_F16,
    )
    return c


import torch


def test_all_branches():
    M, N, K = 128, 128, 128
    NUM_CTAS = 1
    NUM_WARPS = 4

    results = {}

    out = warpper_tma_load_store(
        M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A=False, TRANS_B=False, OUTPUT_F16=False
    )
    results["test_case_1"] = out

    out = warpper_tma_load_store(
        M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A=True, TRANS_B=False, OUTPUT_F16=False
    )
    results["test_case_2"] = out

    out = warpper_tma_load_store(
        M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A=False, TRANS_B=True, OUTPUT_F16=False
    )
    results["test_case_3"] = out

    out = warpper_tma_load_store(
        M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A=True, TRANS_B=True, OUTPUT_F16=False
    )
    results["test_case_4"] = out

    out = warpper_tma_load_store(
        M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A=False, TRANS_B=False, OUTPUT_F16=True
    )
    results["test_case_5"] = out

    out = warpper_tma_load_store(
        M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A=True, TRANS_B=False, OUTPUT_F16=True
    )
    results["test_case_6"] = out

    out = warpper_tma_load_store(
        M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A=False, TRANS_B=True, OUTPUT_F16=True
    )
    results["test_case_7"] = out

    out = warpper_tma_load_store(
        M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A=True, TRANS_B=True, OUTPUT_F16=True
    )
    results["test_case_8"] = out

    return results


result_gold = test_all_branches()

print(result_gold)
