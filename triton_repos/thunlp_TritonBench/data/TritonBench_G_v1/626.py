import torch
import triton
import triton.language as tl


@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def call_kernel(x):

    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


import torch


def test_call_kernel():
    results = {}

    x1 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).cuda()
    output1 = call_kernel(x1)
    results["test_case_1"] = output1

    x2 = torch.linspace(0, 10, steps=1024, dtype=torch.float32).cuda()
    output2 = call_kernel(x2)
    results["test_case_2"] = output2

    x3 = torch.tensor([], dtype=torch.float32).cuda()
    output3 = call_kernel(x3)
    results["test_case_3"] = output3

    x4 = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float32).cuda()
    output4 = call_kernel(x4)
    results["test_case_4"] = output4

    return results


result_gold = test_call_kernel()
