from typing import Any

import torch
import torch.autograd as autograd
import torch.nn as nn
import triton
import triton.language as tl


class ReLUKernel:
    @staticmethod
    @triton.jit
    def forward(output_ptr, input_ptr, size, block_size: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid * block_size

        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(block_size,),
            order=(0,),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(block_size,),
            order=(0,),
        )

        input = tl.load(input_block_ptr, boundary_check=(0,))
        condition = input >= 0
        output = tl.where(condition, input, 0)
        tl.store(output_block_ptr, output, boundary_check=(0,))

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr, grad_output_ptr, input_ptr, size, block_size: tl.constexpr
    ):
        pid = tl.program_id(0)
        offset = pid * block_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(block_size,),
            order=(0,),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(block_size,),
            order=(0,),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(block_size,),
            order=(0,),
        )

        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,))
        input = tl.load(input_block_ptr, boundary_check=(0,))
        condition = input >= 0
        grad_input = tl.where(condition, grad_output, 0)
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(0,))


class ReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        (input,) = args
        output = torch.empty_like(input)
        size = input.numel()
        block_size = triton.next_power_of_2(input.shape[-1])

        def grid(meta):
            return (triton.cdiv(size, meta["block_size"]),)

        ReLUKernel.forward[grid](output, input, size, block_size)

        ctx.save_for_backward(input)

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_output,) = grad_outputs
        (input,) = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)
        size = grad_input.numel()
        block_size = triton.next_power_of_2(grad_input.shape[-1])

        def grid(meta):
            return (triton.cdiv(size, meta["block_size"]),)

        ReLUKernel.backward[grid](grad_input, grad_output, input, size, block_size)

        return grad_input


def relu(input):
    return ReLUFunction.apply(input)


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return relu(input)


def main():
    input = torch.rand(6, device="cuda") * 2 - 1

    input_a = input.clone()
    input_a.requires_grad = True
    grad_output_a = torch.ones_like(input_a)
    output_a = relu(input_a)
    output_a.backward(grad_output_a)

    input_b = input.clone()
    input_b.requires_grad = True
    grad_output_b = torch.ones_like(input_b)
    output_b = torch.nn.functional.relu(input_b)
    output_b.backward(grad_output_b)

    print(f"input ⬇️\ntriton: {input_a.data}\ntorch : {input_b.data}\n")
    print(f"output ⬇️\ntriton: {output_a.data}\ntorch : {output_b.data}\n")
    print(f"input_grad ⬇️\ntriton: {input_a.grad.data}\ntorch : {input_b.grad.data}\n")

    assert torch.allclose(input_a, input_b)
    assert torch.allclose(output_a, output_b)
    assert torch.allclose(input_a.grad, input_b.grad)


if __name__ == "__main__":
    main()
