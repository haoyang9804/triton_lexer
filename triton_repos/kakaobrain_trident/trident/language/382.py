import triton
import triton.language as tl

from trident import language


class Mean:
    @staticmethod
    @triton.jit
    def forward(
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        y_offset: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        sum = tl.zeros((1, x_block_size), dtype)

        for _ in range(0, x_size, x_block_size):
            if require_x_boundary_check:
                input = tl.load(
                    input_block_ptr, boundary_check=(1,), padding_option="zero"
                )
            else:
                input = tl.load(input_block_ptr)

            sum += input
            input_block_ptr = tl.advance(input_block_ptr, (0, x_block_size))

        sum = tl.sum(sum, 1)
        output = sum / x_size

        return output.to(dtype)

    @staticmethod
    @triton.jit
    def backward(
        grad_output_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_offset: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, 1),
            strides=(1, 0),
            offsets=(y_offset, 0),
            block_shape=(1, 1),
            order=(0, 1),
        )

        grad_output = tl.load(grad_output_block_ptr)
        grad_input = tl.broadcast_to(grad_output * 1.0 / x_size, (1, x_block_size))

        return grad_input.to(dtype)
