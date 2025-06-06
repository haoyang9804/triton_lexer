import triton
import triton.language as tl


class GELU:
    @staticmethod
    @triton.jit
    def forward(input: tl.tensor):
        return (
            0.5
            * input
            * (
                1
                + tl.math.tanh(
                    input * 0.7978845608028654 * (1 + 0.044715 * input * input)
                )
            )
        )

    @staticmethod
    @triton.jit
    def backward(grad_output: tl.tensor, input: tl.tensor):
        squared_input = input * input
        alpha = tl.math.tanh(
            0.797884560802865 * (input + 0.044715 * squared_input * input)
        )
        beta = (
            input
            * (1.0 - alpha * alpha)
            * (0.797884560802865 + 0.1070322244089 * squared_input)
        )
        return grad_output * 0.5 * (1.0 + alpha + beta)
