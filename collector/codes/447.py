import triton
import triton.language as tl
import math

sqrt2 = math.sqrt(2.0)


@triton.jit
def relu(x):

    return tl.maximum(0, x)


@triton.jit
def leaky_relu(x):

    scale = 1e-2
    scale = scale.to(x.dtype)
    return tl.where(x >= 0, x, scale * x)


@triton.jit
def tanh(x):

    return 2 / (1 + tl.exp(-2 * x)) - 1


@triton.jit
def gelu(x):

    return x * 0.5 * (1.0 + tl.libdevice.erf(x / sqrt2))


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)
