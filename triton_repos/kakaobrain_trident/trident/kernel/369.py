import triton
import triton.language as tl

from trident import language, util


def relu_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_stages in [4, 5]:
            config = triton.Config(
                {"x_block_size": x_block_size},
                2 if x_block_size <= 512 else 4,
                num_stages,
            )
            configs.append(config)
    return configs


class ReLU:
    @staticmethod
    @util.autotune(relu_configs(), ["x_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        x_size: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        x_offset = tl.program_id(0) * x_block_size
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
        input = tl.load(input_block_ptr, boundary_check=(0,))
        output = language.math.ReLU.forward(input)
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(0,))

    @staticmethod
    @util.autotune(relu_configs(), ["x_size"])
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        x_size: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        x_offset = tl.program_id(0) * x_block_size
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
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,))
        input = tl.load(input_block_ptr, boundary_check=(0,))
        grad_input = language.math.ReLU.backward(grad_output, input)
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0,))
