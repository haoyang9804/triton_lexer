import triton
import triton.language as tl

from trident import language, util


def dropout_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_warps in [2, 4, 8]:
            config = triton.Config({"x_block_size": x_block_size}, num_warps, 4)
            configs.append(config)
    return configs


class Dropout:
    @staticmethod
    @util.autotune(dropout_configs(), ["x_size"])
    @triton.heuristics(
        {"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]}
    )
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        x_size: tl.int32,
        p: tl.float32,
        seed: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x_offset = pid * x_block_size

        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )

        if require_x_boundary_check:
            input = tl.load(input_block_ptr, boundary_check=(0,))
        else:
            input = tl.load(input_block_ptr)

        condition = tl.rand(seed, tl.arange(0, x_block_size) + x_offset) > p
        output = tl.where(condition, input / (1.0 - p + language.eps), 0.0)

        if require_x_boundary_check:
            tl.store(output_block_ptr, output.to(dtype), boundary_check=(0,))
        else:
            tl.store(output_block_ptr, output.to(dtype))

    @staticmethod
    @util.autotune(dropout_configs(), ["x_size"])
    @triton.heuristics(
        {"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]}
    )
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        output_ptr: tl.tensor,
        x_size: tl.int32,
        p: tl.float32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x_offset = pid * x_block_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )

        if require_x_boundary_check:
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,))
            output = tl.load(output_block_ptr, boundary_check=(0,))
        else:
            grad_output = tl.load(grad_output_block_ptr)
            output = tl.load(output_block_ptr)

        condition = (p == 0.0) | (output > 0.0)
        grad_input = tl.where(condition, grad_output * (1.0 - p + language.eps), 0.0)

        if require_x_boundary_check:
            tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0,))
        else:
            tl.store(grad_input_block_ptr, grad_input.to(dtype))
