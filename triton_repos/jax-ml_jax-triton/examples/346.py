import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    block_size: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < 8
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    block_size = 8
    grid = (triton.cdiv(x.size, block_size),)
    return jt.triton_call(
        x, y, kernel=add_kernel, out_shape=out_shape, grid=grid, block_size=block_size
    )


def main(unused_argv):
    x_val = jnp.arange(8)
    y_val = jnp.arange(8, 16)
    print(add(x_val, y_val))
    print(jax.jit(add)(x_val, y_val))


if __name__ == "__main__":
    from absl import app

    app.run(main)
