from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    length,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < length

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def tanh_kernel(
    x_ptr,
    length,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < length

    x = tl.load(x_ptr + offsets, mask=mask)
    output = libdevice.tanh(x)

    tl.store(output_ptr + offsets, output, mask=mask)


class TritonTest(absltest.TestCase):

    def test_add_kernel(self):

        def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
            grid = lambda meta: (triton.cdiv(x.size, meta["BLOCK_SIZE"]),)
            return jt.triton_call(
                x,
                y,
                x.size,
                kernel=add_kernel,
                out_shape=out_shape,
                grid=grid,
                BLOCK_SIZE=8,
            )

        x = jnp.arange(8, dtype=jnp.float32)
        y = jnp.arange(8, dtype=jnp.float32)
        np.testing.assert_allclose(add(x, y), x + y)

    def test_tanh_kernel(self):

        def tanh(x: jnp.ndarray) -> jnp.ndarray:
            out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
            grid = lambda meta: (triton.cdiv(x.size, meta["BLOCK_SIZE"]),)
            return jt.triton_call(
                x,
                x.size,
                kernel=tanh_kernel,
                out_shape=out_shape,
                grid=grid,
                BLOCK_SIZE=8,
            )

        x = jnp.arange(8, dtype=jnp.float32)
        np.testing.assert_allclose(tanh(x), np.tanh(x))


if __name__ == "__main__":
    absltest.main()
