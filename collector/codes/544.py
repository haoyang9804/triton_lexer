from typing import Any

import torch
import triton

from trident import kernel, util


class GELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        (input,) = args

        util.push_trace("GELU.__forward")
        output = GELU.__forward(input)
        util.pop_trace()

        ctx.save_for_backward(input)

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = grad_outputs
        (input,) = ctx.saved_tensors

        util.push_trace("GELU.__backward")
        grad_input = GELU.__backward(grad_output, input)
        util.pop_trace()

        return grad_input

    @staticmethod
    def __forward(input: torch.Tensor):
        x_size = input.numel()
        output = torch.empty_like(input)

        def grid(meta):
            return (triton.cdiv(x_size, meta["x_block_size"]),)

        util.push_trace("kernel.GELU.forward")
        kernel.GELU.forward[grid](output, input, x_size, util.dtype(output.dtype))
        util.pop_trace()

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor):
        x_size = input.numel()
        grad_input = torch.empty_like(input)

        def grid(meta):
            return [triton.cdiv(x_size, meta["x_block_size"])]

        util.push_trace("kernel.GELU.backward")
        kernel.GELU.backward[grid](
            grad_input, grad_output, input, x_size, util.dtype(grad_input.dtype)
        )
        util.pop_trace()

        return grad_input
