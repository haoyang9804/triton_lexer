import triton
import triton.language as tl

from trident import language


class VarMean:
    @staticmethod
    @triton.jit
    def forward(
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        y_offset: tl.int32,
        correction: tl.constexpr,
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

        m2 = tl.zeros((1, x_block_size), tl.float32)
        count = tl.zeros((1, x_block_size), tl.float32)
        mean = tl.zeros((1, x_block_size), tl.float32)

        for x_offset in range(0, x_size, x_block_size):
            if require_x_boundary_check:
                input = tl.load(input_block_ptr, boundary_check=(1,))
                condition = tl.arange(0, x_block_size) + x_offset < x_size
                delta = tl.where(condition, input - mean, 0.0)
                count += tl.where(condition, 1.0, language.eps)
                mean += delta / count
                m2 += delta * tl.where(condition, input - mean, 0.0)
            else:
                input = tl.load(input_block_ptr)
                delta = input - mean
                count += 1
                mean += delta / count
                m2 += delta * (input - mean)

            input_block_ptr = tl.advance(input_block_ptr, (0, x_block_size))

        m2, mean, count = tl.reduce((m2, mean, count), 1, language.combine_welford)
        output = m2 / (x_size - correction)

        return output.to(dtype), mean.to(dtype)

    @staticmethod
    @triton.jit
    def backward(
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        y_offset: tl.int32,
        x_offset: tl.int32,
        mean: tl.tensor,
        correction: tl.constexpr,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, 1),
            strides=(1, 0),
            offsets=(y_offset, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        if require_x_boundary_check:
            input = tl.load(input_block_ptr, boundary_check=(1,))
            condition = tl.arange(0, x_block_size) + x_offset < x_size
            centered_mean = tl.where(condition[None, :], input - mean, 0.0)
        else:
            input = tl.load(input_block_ptr)
            centered_mean = input - mean

        grad_output = tl.load(grad_output_block_ptr)
        grad_input = grad_output * 2 * centered_mean / (x_size - correction)

        return grad_input.to(dtype)
