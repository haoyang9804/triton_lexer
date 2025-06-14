import triton
import triton.language as tl

from trident import language, util


def softmax_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_warps in [4, 8, 16]:
            configs.append(
                triton.Config({"x_block_size": x_block_size}, num_warps=num_warps)
            )
    return configs


class Softmax:
    @staticmethod
    @util.autotune(softmax_configs(), ["x_size"])
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
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        max = tl.full((1, x_block_size), -float("inf"), tl.float32)
        sum = tl.zeros((1, x_block_size), tl.float32)

        for x_offset in range(0, x_size, x_block_size):
            if require_x_boundary_check:
                input = tl.load(input_block_ptr, boundary_check=(1,))
                condition = tl.arange(0, x_block_size) + x_offset < x_size
                input = tl.where(condition, input, -float("inf"))
                peak = tl.where(condition, tl.maximum(max, input), 0)
            else:
                input = tl.load(input_block_ptr)
                peak = tl.maximum(max, input)

            sum = sum * tl.math.fast_expf(max - peak) + tl.math.fast_expf(input - peak)
            max = peak
            input_block_ptr = tl.advance(input_block_ptr, (0, x_block_size))

        max, sum = tl.reduce((max, sum), 1, language.combine_softmax)

        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        for x_offset in range(0, x_size, x_block_size):
            if require_x_boundary_check:
                input = tl.load(input_block_ptr, boundary_check=(1,))
            else:
                input = tl.load(input_block_ptr)

            output = tl.math.fast_expf(input - max) / sum

            if require_x_boundary_check:
                tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
            else:
                tl.store(output_block_ptr, output.to(dtype))

            output_block_ptr = tl.advance(output_block_ptr, (0, x_block_size))
            input_block_ptr = tl.advance(input_block_ptr, (0, x_block_size))

    @staticmethod
    @util.autotune(softmax_configs(), ["x_size"])
    @triton.heuristics(
        {"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]}
    )
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        output_ptr: tl.tensor,
        delta_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        y_offset = tl.program_id(0)

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        delta_block_ptr = tl.make_block_ptr(
            delta_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )

        for x_offset in range(0, x_size, x_block_size):
            if require_x_boundary_check:
                output = tl.load(output_block_ptr, boundary_check=(1,))
                grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
            else:
                output = tl.load(output_block_ptr)
                grad_output = tl.load(grad_output_block_ptr)

            delta = tl.load(delta_block_ptr)
            grad_input = output * (grad_output - delta)

            if require_x_boundary_check:
                tl.store(
                    grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,)
                )
            else:
                tl.store(grad_input_block_ptr, grad_input.to(dtype))

            grad_input_block_ptr = tl.advance(grad_input_block_ptr, (0, x_block_size))
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, x_block_size))
            output_block_ptr = tl.advance(output_block_ptr, (0, x_block_size))

    @staticmethod
    @util.autotune(softmax_configs(), ["x_size"])
    @triton.heuristics(
        {"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]}
    )
    @triton.jit
    def backward_delta(
        delta_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        output_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        y_offset = tl.program_id(0)

        delta_block_ptr = tl.make_block_ptr(
            delta_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        delta = tl.zeros((1, x_block_size), dtype)

        for _ in range(0, x_size, x_block_size):
            if require_x_boundary_check:
                grad_output = tl.load(
                    grad_output_block_ptr, boundary_check=(1,), padding_option="zero"
                )
                output = tl.load(output_block_ptr, boundary_check=(1,))
            else:
                grad_output = tl.load(grad_output_block_ptr)
                output = tl.load(output_block_ptr)

            delta += grad_output * output
            output_block_ptr = tl.advance(output_block_ptr, (0, x_block_size))
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, x_block_size))

        delta = tl.sum(delta, 1)
        tl.store(delta_block_ptr, delta.to(dtype))
