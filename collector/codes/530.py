import triton
import triton.language as tl


class ReLU:
    @staticmethod
    @triton.jit
    def forward(input: tl.tensor):
        return tl.where(input > 0, input, 0)

    @staticmethod
    @triton.jit
    def backward(grad_output: tl.tensor, input: tl.tensor):
        return tl.where(input > 0, grad_output, 0)
