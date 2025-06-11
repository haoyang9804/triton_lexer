import triton
import triton.language as tl


@triton.jit
def clamp(x: tl.tensor, min_val, max_val) -> tl.tensor:

    return tl.math.min(tl.math.max(x, min_val), max_val)


@triton.jit
def dequantize(x: tl.tensor, scale: tl.tensor) -> tl.tensor:

    return (x * scale).to(tl.float32)


@triton.jit
def quantize(x, scale, qmin, qmax) -> tl.tensor:

    return clamp(tl.math.round(x / scale), qmin, qmax)
