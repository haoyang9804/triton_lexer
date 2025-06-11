import torch
import triton
import triton.language as tl
import math


@triton.jit
def cos_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):

    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offset < n_elements

    a_value = tl.load(a + offset, mask=mask)

    b_value = tl.cos(a_value.to(tl.float32))

    tl.store(b + offset, b_value, mask=mask)


def cos(A):

    B = torch.empty_like(A)

    n_elements = A.numel()

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))

    grid_size = triton.cdiv(n_elements, block_size)

    cos_func[(grid_size, 1, 1)](A, B, n_elements, block_size)
    return B


def test_cos_function():

    test_cases = {
        "test_case_1": torch.rand(1024, device="cuda") * 2 * math.pi,
        "test_case_2": torch.rand(2048, device="cuda") * 2 * math.pi,
        "test_case_3": torch.rand(4096, device="cuda") * 2 * math.pi,
        "test_case_4": torch.rand(8192, device="cuda") * 2 * math.pi,
    }

    results = {}

    for case_name, input_tensor in test_cases.items():

        B_triton = cos(input_tensor)
        results[case_name] = B_triton

    return results


result_gold = test_cos_function()
