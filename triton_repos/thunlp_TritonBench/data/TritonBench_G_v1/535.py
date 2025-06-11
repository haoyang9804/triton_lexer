import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_packed_to_bf16(
    x_packed,
    sign_mask_f4,
    mantissa_mask_f4,
    mbits_f4_e2m1,
    ebits_f4_e2m1,
    f4_e2m1_exp_bias,
    mbits_f32,
    ebits_f32,
    f32_exp_bias,
    zero_bits_f32,
    zero_point_five_bits_f32,
):

    x_low_bits = x_packed >> 4
    x_high_bits = x_packed & 0xF
    x = tl.interleave(x_low_bits, x_high_bits)

    sign_f4 = x & sign_mask_f4

    x_pos = x ^ sign_f4

    zero_mask = x_pos == 0

    denormal_mask = x_pos == 1

    exp_biased_f4 = x_pos >> mbits_f4_e2m1
    exp_biased_f32 = exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias
    exp_biased_f32 = exp_biased_f32.to(tl.int32) << mbits_f32

    mantissa_f4 = x_pos & mantissa_mask_f4
    mantissa_f32 = mantissa_f4.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)
    output = mantissa_f32

    result = exp_biased_f32 | mantissa_f32

    result = tl.where(zero_mask, zero_bits_f32, result)

    result = tl.where(denormal_mask, zero_point_five_bits_f32, result)

    sign_f32 = sign_f4.to(tl.int32) << (
        mbits_f32 - mbits_f4_e2m1 + ebits_f32 - ebits_f4_e2m1
    )
    result = result | sign_f32

    output = result.to(tl.float32, bitcast=True)
    output = output.to(tl.bfloat16)
    return output


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_IN": 128}),
        triton.Config({"BLOCK_SIZE_IN": 256}),
        triton.Config({"BLOCK_SIZE_IN": 512}),
        triton.Config({"BLOCK_SIZE_IN": 1024}),
        triton.Config({"BLOCK_SIZE_IN": 2048}),
    ],
    key=["n_elements_in"],
)
@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,
    s_ptr,
    output_ptr,
    n_elements_in,
    mx_block_size: tl.constexpr,
    sign_mask_f4: tl.constexpr,
    mantissa_mask_f4: tl.constexpr,
    mbits_f4_e2m1: tl.constexpr,
    ebits_f4_e2m1: tl.constexpr,
    f4_e2m1_exp_bias: tl.constexpr,
    mbits_f32: tl.constexpr,
    ebits_f32: tl.constexpr,
    f32_exp_bias: tl.constexpr,
    zero_bits_f32: tl.constexpr,
    zero_point_five_bits_f32: tl.constexpr,
    e8m0_exponent_bias: tl.constexpr,
    e8m0_exponent_nan_val: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_elements_out = n_elements_in * 2
    n_elements_s = n_elements_out // 32

    BLOCK_SIZE_S: tl.constexpr = BLOCK_SIZE_IN // 16
    BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2

    block_start_in = pid * BLOCK_SIZE_IN
    offsets_in = block_start_in + tl.arange(0, BLOCK_SIZE_IN)
    mask_in = offsets_in < n_elements_in

    x_packed = tl.load(x_ptr + offsets_in, mask=mask_in)
    output = _fp4_packed_to_bf16(
        x_packed,
        sign_mask_f4,
        mantissa_mask_f4,
        mbits_f4_e2m1,
        ebits_f4_e2m1,
        f4_e2m1_exp_bias,
        mbits_f32,
        ebits_f32,
        f32_exp_bias,
        zero_bits_f32,
        zero_point_five_bits_f32,
    )

    block_start_s = pid * BLOCK_SIZE_S
    offsets_s = block_start_s + tl.arange(0, BLOCK_SIZE_S)
    mask_s = offsets_s < n_elements_s
    s = tl.load(s_ptr + offsets_s, mask=mask_s)

    s_offset = s.to(tl.int16) - e8m0_exponent_bias
    s_fp = tl.extra.cuda.libdevice.pow(2.0, s_offset).to(tl.bfloat16)
    s_fp = tl.where(s != e8m0_exponent_nan_val, s_fp, float("nan"))

    output = tl.reshape(output, (BLOCK_SIZE_OUT // mx_block_size, mx_block_size))
    s_fp = tl.reshape(s_fp, (BLOCK_SIZE_S // 1, 1))
    output = output * s_fp
    output = tl.reshape(output, (BLOCK_SIZE_OUT,))

    block_start_out = pid * BLOCK_SIZE_OUT
    offsets_out = block_start_out + tl.arange(0, BLOCK_SIZE_OUT)
    mask_out = offsets_out < n_elements_out

    tl.store(output_ptr + offsets_out, output, mask=mask_out)


EBITS_F32, MBITS_F32 = 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2

SIGN_MASK_F4 = 0x8
MANTISSA_MASK_F4 = 0x1

ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000
F4_E2M1_EXP_BIAS = 1
F32_EXP_BIAS = 127
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255


def triton_f4_to_scaled_bf16(
    x: torch.Tensor,
    s_e8m0: torch.Tensor,
    mx_block_size: int,
):

    new_shape = (*x.shape[:-1], x.shape[-1] * 2)
    output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)
    assert x.is_contiguous()
    assert x.is_cuda and output.is_cuda
    n_elements_in = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements_in, meta["BLOCK_SIZE_IN"]),)
    triton_f4_to_scaled_bf16_kernel[grid](
        x,
        s_e8m0,
        output,
        n_elements_in,
        mx_block_size,
        sign_mask_f4=SIGN_MASK_F4,
        mantissa_mask_f4=MANTISSA_MASK_F4,
        mbits_f4_e2m1=MBITS_F4_E2M1,
        ebits_f4_e2m1=EBITS_F4_E2M1,
        f4_e2m1_exp_bias=F4_E2M1_EXP_BIAS,
        mbits_f32=MBITS_F32,
        ebits_f32=EBITS_F32,
        f32_exp_bias=F32_EXP_BIAS,
        zero_bits_f32=ZERO_BITS_F32,
        zero_point_five_bits_f32=ZERO_POINT_FIVE_BITS_F32,
        e8m0_exponent_bias=E8M0_EXPONENT_BIAS,
        e8m0_exponent_nan_val=E8M0_EXPONENT_NAN_VAL,
    )
    return output


def test_triton_f4_to_scaled_bf16():
    device = "cuda"
    mx_block_size = 32
    n_elements_in = 1024

    x = torch.randint(0, 256, (n_elements_in,), dtype=torch.uint8, device=device)

    n_elements_out = n_elements_in * 2
    n_elements_s = n_elements_out // 32

    s_e8m0 = torch.randint(0, 256, (n_elements_s,), dtype=torch.uint8, device=device)

    output1 = triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size)

    output2 = triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size)

    output3 = triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size)

    output4 = triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size)

    output5 = triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size)

    results = {
        "test_case_1": output1,
        "test_case_2": output2,
        "test_case_3": output3,
        "test_case_4": output4,
        "test_case_5": output5,
    }

    return results


result_gold = test_triton_f4_to_scaled_bf16()
