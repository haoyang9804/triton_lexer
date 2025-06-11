import triton
import triton.language as tl


@triton.jit
def gelu(input):

    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input
