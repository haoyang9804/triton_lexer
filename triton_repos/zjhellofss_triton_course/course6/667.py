import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    sigmoid_x_fp32 = 1 / (1 + tl.exp(-x_fp32))
    sigmoid_x = sigmoid_x_fp32.to(tl.float16)
    output = x * sigmoid_x.to(tl.float16)
    tl.store(output_ptr + offsets, output, mask=mask)


def silu(x: torch.Tensor):
    output = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


if __name__ == "__main__":
    x = torch.randn(10000, device="cuda").half()
    output_triton = silu(x)
    output_torch = torch.nn.functional.silu(x)
    print("最大差异:", torch.max(torch.abs(output_triton - output_torch)))
