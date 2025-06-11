import torch
import triton
import triton.language as tl


@triton.jit
def mul_kernel(src, dst, BLOCK_SIZE: tl.constexpr):

    exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)

    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(src + idxs)

    y = x * exponent_compensator

    tl.store(dst + idxs, y)


def launch_mul_kernel(src, BLOCK_SIZE=1):

    dst = torch.empty(src.shape, dtype=torch.float32, device="cuda")

    mul_kernel[(src.shape[0] // BLOCK_SIZE,)](src, dst, BLOCK_SIZE)
    return dst


def test_mul():
    src = torch.tensor([8323072], dtype=torch.int32, device="cuda").view(torch.float32)

    test_cases = {}

    dst_triton_1 = launch_mul_kernel(src, BLOCK_SIZE=1)
    test_cases["test_case_1"] = dst_triton_1

    dst_triton_2 = launch_mul_kernel(src, BLOCK_SIZE=2)
    test_cases["test_case_2"] = dst_triton_2

    dst_triton_3 = launch_mul_kernel(src, BLOCK_SIZE=4)
    test_cases["test_case_3"] = dst_triton_3

    dst_triton_4 = launch_mul_kernel(src, BLOCK_SIZE=8)
    test_cases["test_case_4"] = dst_triton_4

    return test_cases


result_gold = test_mul()
