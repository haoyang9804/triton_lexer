import triton
import triton.language as tl

from trident import language, util


def mean_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_stages in [4, 5]:
            config = triton.Config({"x_block_size": x_block_size}, 8, num_stages)
            configs.append(config)
    return configs


class Mean:
    @staticmethod
    @util.autotune(mean_configs(), ["x_size"])
    @triton.heuristics(
        {"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]}
    )
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        y_offset = tl.program_id(0)

        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )

        output = language.Mean.forward(
            input_ptr,
            y_size,
            x_size,
            y_stride,
            x_stride,
            y_offset,
            dtype,
            x_block_size,
            require_x_boundary_check,
        )
        tl.store(output_block_ptr, output)

    @staticmethod
    @util.autotune(mean_configs(), ["x_size"])
    @triton.heuristics(
        {"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]}
    )
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        y_offset = pid // num_x_blocks
        x = pid % num_x_blocks
        x_offset = x * x_block_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(0, 1),
        )

        grad_input = language.Mean.backward(
            grad_output_ptr, y_size, x_size, y_offset, dtype, x_block_size
        )

        if require_x_boundary_check:
            tl.store(grad_input_block_ptr, grad_input, boundary_check=(1,))
        else:
            tl.store(grad_input_block_ptr, grad_input)
