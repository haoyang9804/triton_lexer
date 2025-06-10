import triton
import triton.language as tl


class LeakyReLU:
    @staticmethod
    @triton.jit
    def forward(input: tl.tensor, negative_slope: tl.float32):
        condition = input > 0
        return tl.where(condition, input, 0) + negative_slope * tl.where(
            condition, 0, input
        )

    @staticmethod
    @triton.jit
    def backward(grad_output: tl.tensor, input: tl.tensor, negative_slope: tl.float32):
        return grad_output * tl.where(input > 0, 1, negative_slope)
