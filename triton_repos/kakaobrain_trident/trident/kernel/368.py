import triton
import triton.language as tl

from trident import language, util


def prelu_configs():
    configs = []
    for y_block_size in [32, 64, 128, 256]:
        for x_block_size in [1, 32, 64]:
            for num_warps in [4, 8, 16]:
                config = triton.Config(
                    {"y_block_size": y_block_size, "x_block_size": x_block_size},
                    num_warps,
                )
                configs.append(config)
    return configs


class PReLU:
    @staticmethod
    @util.autotune(prelu_configs(), ["x_size"])
    @triton.heuristics(
        {
            "require_y_boundary_check": lambda args: args["y_size"]
            % args["y_block_size"],
            "require_x_boundary_check": lambda args: args["x_size"]
            % args["x_block_size"],
        }
    )
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        num_batches: tl.int32,
        y_size: tl.int32,
        x_size: tl.int32,
        batch_stride: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        y_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
        require_y_boundary_check: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_y_blocks = tl.cdiv(y_size, y_block_size)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        num_blocks = num_y_blocks * num_x_blocks
        batch_offset = pid // num_blocks
        block = pid % num_blocks
        y_block = block // num_x_blocks
        x_block = block % num_x_blocks
        y_offset = y_block * y_block_size
        x_offset = x_block * x_block_size

        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(y_size, 1),
            strides=(1, 0),
            offsets=(y_offset, 0),
            block_shape=(y_block_size, 1),
            order=(1, 0),
        )

        if require_y_boundary_check | require_x_boundary_check:
            input = tl.load(input_block_ptr, boundary_check=(1, 2))
        else:
            input = tl.load(input_block_ptr)

        if require_y_boundary_check:
            weight = tl.load(weight_block_ptr, boundary_check=(0,))
        else:
            weight = tl.load(weight_block_ptr)

        output = language.math.LeakyReLU.forward(input, weight)

        if require_y_boundary_check | require_x_boundary_check:
            tl.store(output_block_ptr, output.to(dtype), boundary_check=(1, 2))
        else:
            tl.store(output_block_ptr, output.to(dtype))

    @staticmethod
    @util.autotune(prelu_configs(), ["x_size"])
    @triton.heuristics(
        {
            "require_y_boundary_check": lambda args: args["y_size"]
            % args["y_block_size"],
            "require_x_boundary_check": lambda args: args["x_size"]
            % args["x_block_size"],
        }
    )
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_weight_staging_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        num_batches: tl.int32,
        y_size: tl.int32,
        x_size: tl.int32,
        batch_stride: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        y_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
        require_y_boundary_check: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_y_blocks = tl.cdiv(y_size, y_block_size)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        num_blocks = num_y_blocks * num_x_blocks
        batch_offset = pid // num_blocks
        block = pid % num_blocks
        y_block = block // num_x_blocks
        x_block = block % num_x_blocks
        y_offset = y_block * y_block_size
        x_offset = x_block * x_block_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        grad_weight_staging_block_ptr = tl.make_block_ptr(
            grad_weight_staging_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(y_size, 1),
            strides=(1, 0),
            offsets=(y_offset, 0),
            block_shape=(y_block_size, 1),
            order=(1, 0),
        )
        if require_y_boundary_check | require_x_boundary_check:
            input = tl.load(input_block_ptr, boundary_check=(1, 2))
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1, 2))
        else:
            input = tl.load(input_block_ptr)
            grad_output = tl.load(grad_output_block_ptr)

        weight = tl.load(weight_block_ptr)
        grad_input = language.math.LeakyReLU.backward(grad_output, input, weight)
        grad_weight = grad_output * tl.where(input > 0, 0, input)

        if require_y_boundary_check | require_x_boundary_check:
            tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1, 2))
            tl.store(
                grad_weight_staging_block_ptr,
                grad_weight.to(dtype),
                boundary_check=(1, 2),
            )
        else:
            tl.store(grad_input_block_ptr, grad_input.to(dtype))
            tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype))
