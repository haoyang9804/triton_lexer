import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=16),
    ],
    key=["M", "N"],
)
@triton.jit
def quantize_2d_bf16_to_int2(
    bf16_ptr,
    int8_ptr,
    M,
    N,
    stride_bm,
    stride_bn,
    stride_im,
    stride_in,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)

    mask_m = offs_m < M
    mask_n = offs_n < (N // 4)

    packed_output = tl.zeros((BLOCK_M, BLOCK_N // 4), dtype=tl.int32)

    for i in range(4):

        column_idx = offs_n * 4 + i
        bf16_vals = tl.load(
            bf16_ptr + offs_m[:, None] * stride_bm + column_idx[None, :] * stride_bn,
            mask=mask_m[:, None] & (column_idx[None, :] < N),
            other=0.0,
        )

        int2_vals = tl.where(
            bf16_vals == -2.0,
            0,
            tl.where(
                bf16_vals == -1.0,
                1,
                tl.where(bf16_vals == 0.0, 2, 3),
            ),
        )

        packed_output = packed_output | (int2_vals << (i * 2))

    tl.store(
        int8_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in,
        packed_output.to(tl.int8),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def quantize_bf16_to_int2(input_tensor):

    assert (
        input_tensor.dtype == torch.bfloat16
    ), "Input tensor must be of dtype torch.bfloat16"
    assert input_tensor.dim() == 2, "Input tensor must be 2D"
    M, K = input_tensor.shape
    assert (
        K % 4 == 0
    ), "Input tensor's second dimension (K) must be divisible by 4 for INT2 packing"

    output_tensor = torch.empty(
        (M, K // 4), dtype=torch.int8, device=input_tensor.device
    )

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(K, meta["BLOCK_N"]),
    )

    quantize_2d_bf16_to_int2[grid](
        bf16_ptr=input_tensor,
        int8_ptr=output_tensor,
        M=M,
        N=K,
        stride_bm=input_tensor.stride(0),
        stride_bn=input_tensor.stride(1),
        stride_im=output_tensor.stride(0),
        stride_in=output_tensor.stride(1),
    )

    return output_tensor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=16),
    ],
    key=["M", "N"],
)
@triton.jit
def dequantize_2d_int2_to_bf16(
    int8_ptr,
    bf16_ptr,
    M,
    N,
    stride_im,
    stride_in,
    stride_bm,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)

    mask_m = offs_m < M
    mask_n = offs_n < (N // 4)

    packed_vals = tl.load(
        int8_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0,
    ).to(tl.int32)

    for i in range(4):
        shift = i * 2
        mask = 0b11 << shift
        int2_vals = (packed_vals & mask) >> shift

        bf16_vals = tl.where(
            int2_vals == 0b00,
            tl.full(int2_vals.shape, -2.0, dtype=tl.float32),
            tl.where(
                int2_vals == 0b01,
                tl.full(int2_vals.shape, -1.0, dtype=tl.float32),
                tl.where(
                    int2_vals == 0b10,
                    tl.full(int2_vals.shape, 0.0, dtype=tl.float32),
                    tl.full(int2_vals.shape, 1.0, dtype=tl.float32),
                ),
            ),
        )

        output_idx = offs_n * 4 + i

        tl.store(
            bf16_ptr + offs_m[:, None] * stride_bm + output_idx[None, :] * stride_bn,
            bf16_vals.to(tl.bfloat16),
            mask=mask_m[:, None] & (output_idx[None, :] < N),
        )


def dequantize_int2_to_bf16(input_tensor):

    assert input_tensor.dtype == torch.int8, "Input tensor must be of dtype torch.int8"
    assert input_tensor.dim() == 2, "Input tensor must be 2D"
    M, K_packed = input_tensor.shape
    K = K_packed * 4

    output_tensor = torch.empty(
        (M, K), dtype=torch.bfloat16, device=input_tensor.device
    )

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(K_packed * 4, meta["BLOCK_N"]),
    )

    dequantize_2d_int2_to_bf16[grid](
        int8_ptr=input_tensor,
        bf16_ptr=output_tensor,
        M=M,
        N=K,
        stride_im=input_tensor.stride(0),
        stride_in=input_tensor.stride(1),
        stride_bm=output_tensor.stride(0),
        stride_bn=output_tensor.stride(1),
    )

    return output_tensor


if __name__ == "__main__":

    M, K = 512, 128
    bf16_tensor = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0] * (K // 4) for _ in range(M)],
        dtype=torch.bfloat16,
        device="cuda",
    )

    int8_packed_tensor = quantize_bf16_to_int2(bf16_tensor)
    bf16_unpacked_tensor = dequantize_int2_to_bf16(int8_packed_tensor)

    print("Input BF16 tensor:", bf16_tensor)
    print("Packed INT8 tensor:", int8_packed_tensor)
    print("Unpacked BF16 tensor:", bf16_unpacked_tensor)

    torch.testing.assert_close(bf16_tensor, bf16_unpacked_tensor)
