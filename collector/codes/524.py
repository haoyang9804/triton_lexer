import triton
import triton.language as tl


@triton.jit
def dot(
    input: tl.tensor,
    other: tl.tensor,
    use_accelerator: tl.constexpr,
    dtype: tl.constexpr,
):
    if dtype is tl.bfloat16:
        return tl.dot(
            input, other, allow_tf32=use_accelerator, out_dtype=tl.float16
        ).to(dtype)
    else:
        return tl.dot(input, other, allow_tf32=use_accelerator, out_dtype=dtype)
