import triton
import triton.language as tl

from trident import language


class Linear:
    @staticmethod
    @triton.jit
    def forward(
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        input_m_stride: tl.int32,
        input_k_stride: tl.int32,
        weight_n_stride: tl.int32,
        weight_k_stride: tl.int32,
        m_offset: tl.int32,
        n_offset: tl.int32,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        require_m_boundary_check: tl.constexpr,
        require_n_boundary_check: tl.constexpr,
        require_k_boundary_check: tl.constexpr,
        dtype: tl.constexpr,
    ):
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(m_size, k_size),
            strides=(input_m_stride, input_k_stride),
            offsets=(m_offset, 0),
            block_shape=(m_block_size, k_block_size),
            order=(1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(k_size, n_size),
            strides=(weight_k_stride, weight_n_stride),
            offsets=(0, n_offset),
            block_shape=(k_block_size, n_block_size),
            order=(0, 1),
        )
        output = tl.zeros((m_block_size, n_block_size), dtype)

        for k_offset in range(0, k_size, k_block_size):
            if require_k_boundary_check | require_m_boundary_check:
                input = tl.load(
                    input_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )
            else:
                input = tl.load(input_block_ptr)

            if require_k_boundary_check | require_n_boundary_check:
                weight = tl.load(
                    weight_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )
            else:
                weight = tl.load(weight_block_ptr)

            output += language.dot(input, weight, use_accelerator, dtype)
            input_block_ptr = tl.advance(input_block_ptr, (0, k_block_size))
            weight_block_ptr = tl.advance(weight_block_ptr, (k_block_size, 0))

        if bias_ptr is not None:
            bias_block_ptr = tl.make_block_ptr(
                bias_ptr,
                shape=(n_size,),
                strides=(1,),
                offsets=(n_offset,),
                block_shape=(n_block_size,),
                order=(0,),
            )
            if require_n_boundary_check:
                bias = tl.load(
                    bias_block_ptr, boundary_check=(0,), padding_option="zero"
                )
            else:
                bias = tl.load(bias_block_ptr)

            output += bias

        return output

    @staticmethod
    @triton.jit
    def backward(
        grad_output_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        weight_n_stride: tl.int32,
        weight_k_stride: tl.int32,
        m_offset: tl.int32,
        k_offset: tl.int32,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        require_m_boundary_check: tl.constexpr,
        require_n_boundary_check: tl.constexpr,
        require_k_boundary_check: tl.constexpr,
        dtype: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, 0),
            block_shape=(m_block_size, n_block_size),
            order=(1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(n_size, k_size),
            strides=(weight_n_stride, weight_k_stride),
            offsets=(0, k_offset),
            block_shape=(n_block_size, k_block_size),
            order=(1, 0),
        )
        grad_input = tl.zeros((m_block_size, k_block_size), dtype)

        for _ in range(0, n_size, n_block_size):
            if require_n_boundary_check | require_m_boundary_check:
                grad_output = tl.load(
                    grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )
            else:
                grad_output = tl.load(grad_output_block_ptr)

            if require_n_boundary_check | require_k_boundary_check:
                weight = tl.load(
                    weight_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )
            else:
                weight = tl.load(weight_block_ptr)

            grad_input += language.dot(grad_output, weight, use_accelerator, dtype)
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, n_block_size))
            weight_block_ptr = tl.advance(weight_block_ptr, (n_block_size, 0))

        return grad_input

    @staticmethod
    @triton.jit
    def backward_weight(
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        input_m_stride: tl.int32,
        input_k_stride: tl.int32,
        n_offset: tl.int32,
        k_offset: tl.int32,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        require_m_boundary_check: tl.constexpr,
        require_n_boundary_check: tl.constexpr,
        require_k_boundary_check: tl.constexpr,
        dtype: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(n_size, m_size),
            strides=(1, n_size),
            offsets=(n_offset, 0),
            block_shape=(n_block_size, m_block_size),
            order=(0, 1),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(m_size, k_size),
            strides=(input_m_stride, input_k_stride),
            offsets=(0, k_offset),
            block_shape=(m_block_size, k_block_size),
            order=(1, 0),
        )
        grad_weight = tl.zeros((n_block_size, k_block_size), dtype)

        for _ in range(0, m_size, m_block_size):
            if require_m_boundary_check | require_n_boundary_check:
                grad_output = tl.load(
                    grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )
            else:
                grad_output = tl.load(grad_output_block_ptr)

            if require_m_boundary_check | require_k_boundary_check:
                input = tl.load(
                    input_block_ptr, boundary_check=(0, 1), padding_option="zero"
                )
            else:
                input = tl.load(input_block_ptr)

            grad_weight += language.dot(grad_output, input, use_accelerator, dtype)
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, m_block_size))
            input_block_ptr = tl.advance(input_block_ptr, (m_block_size, 0))

        return grad_weight

    @staticmethod
    @triton.jit
    def backward_bias(
        grad_output_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        n_offset: tl.int32,
        m_block_size: tl.constexpr,
        require_m_boundary_check: tl.constexpr,
        dtype: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(n_size, m_size),
            strides=(1, n_size),
            offsets=(n_offset, 0),
            block_shape=(1, m_block_size),
            order=(0, 1),
        )
        grad_bias = tl.zeros((1, m_block_size), dtype)

        for m_offset in range(0, m_size, m_block_size):
            if require_m_boundary_check:
                grad_output = tl.load(
                    grad_output_block_ptr, boundary_check=(1,), padding_option="zero"
                )
            else:
                grad_output = tl.load(grad_output_block_ptr)

            grad_bias += grad_output
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, m_block_size))

        return tl.sum(grad_bias, 1).to(dtype)
