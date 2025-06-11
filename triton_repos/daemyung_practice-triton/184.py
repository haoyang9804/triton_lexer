import torch
import triton
import triton.language as tl


@triton.heuristics({"boundary_check": lambda args: args["x_size"] % args["block_size"]})
@triton.jit
def add_kernel(
    x_ptr, y_ptr, z_ptr, size, block_size: tl.constexpr, boundary_check: tl.constexpr
):
    offset = tl.program_id(0) * block_size

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offset,),
        block_shape=(block_size,),
        order=(0,),
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offset,),
        block_shape=(block_size,),
        order=(0,),
    )

    if boundary_check:
        x = tl.load(x_block_ptr, boundary_check=(0,))
        y = tl.load(y_block_ptr, boundary_check=(0,))
    else:
        x = tl.load(x_block_ptr)
        y = tl.load(y_block_ptr)

    z = x + y

    z_block_ptr = tl.make_block_ptr(
        z_ptr,
        shape=(size,),
        strides=(1,),
        offsets=(offset,),
        block_shape=(block_size,),
        order=(0,),
    )

    if boundary_check:
        tl.store(z_block_ptr, z, boundary_check=(0,))
    else:
        tl.store(z_block_ptr, z)


def add(x, y):
    z = torch.empty_like(x, device="cuda")
    size = z.numel()

    def grid(meta):
        return (triton.cdiv(size, meta["block_size"]),)

    add_kernel[grid](x, y, z, size, 1024)

    return z


def main():
    size = 2**16
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    a = add(x, y)
    b = x + y

    assert torch.allclose(a, b)


if __name__ == "__main__":
    main()
