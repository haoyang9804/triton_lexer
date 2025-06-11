import math

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

next_pow2 = lambda x: int(math.pow(2, math.ceil(math.log(x, 2))))


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    block_size: tl.constexpr,
):

    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, block_size)
    input_ptrs = row_start_ptr + col_offsets

    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))

    row_minus_max = row - tl.max(row, axis=0)

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x: jnp.ndarray) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    block_size = next_pow2(x.shape[1])
    strides = jt.strides_from_shape(x.shape)
    return jt.triton_call(
        x,
        kernel=softmax_kernel,
        out_shape=out_shape,
        input_row_stride=strides[0],
        output_row_stride=strides[0],
        n_cols=x.shape[1],
        grid=x.shape[0],
        block_size=block_size,
    )


def main(unused_argv):
    x_val = jnp.ones((8, 5), dtype="float32")
    print(softmax(x_val).block_until_ready())
    print(jax.jit(softmax)(x_val).block_until_ready())


if __name__ == "__main__":
    from absl import app

    app.run(main)
