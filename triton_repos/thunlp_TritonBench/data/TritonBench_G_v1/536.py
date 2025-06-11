import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,
    bias_ptr,
    in_ptr,
    num_weights: tl.constexpr,
    xnumel: tl.constexpr,
    multiplier: tl.constexpr,
    activation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    index = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < xnumel
    bias_index = index % num_weights
    tmp0 = tl.load(x_ptr + index, mask)
    tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy="evict_last")
    tmp3 = tl.load(in_ptr + index, mask)
    activ_input = multiplier * tmp3 + tmp0 + tmp1
    if activation == "sigmoid":
        ma_result = tl.sigmoid(activ_input)

    elif activation == "relu":
        ma_result = tl.maximum(0, activ_input)

    tl.store(x_ptr + index, ma_result, mask)


def fused_add_mul_activation_torch(
    in_out_tensor: torch.Tensor, bias: torch.Tensor, in_tensor: torch.Tensor
) -> torch.Tensor:

    grid = lambda meta: (triton.cdiv(in_out_tensor.numel(), meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = min(2048, in_out_tensor.numel())
    fused_add_mul_activation_kernel[grid](
        in_out_tensor,
        bias,
        in_tensor,
        bias.numel(),
        in_out_tensor.numel(),
        multiplier=0.5,
        activation="sigmoid",
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return in_out_tensor


def test_fused_add_mul_activation():

    num_elements = 8192
    num_weights = 64

    in_out_tensor = torch.randn(num_elements, dtype=torch.float32, device="cuda")
    bias = torch.randn(num_weights, dtype=torch.float32, device="cuda")
    in_tensor = torch.randn(num_elements, dtype=torch.float32, device="cuda")

    result_sigmoid = fused_add_mul_activation_torch(
        in_out_tensor.clone(), bias, in_tensor
    )

    grid = lambda meta: (triton.cdiv(in_out_tensor.numel(), meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = min(2048, in_out_tensor.numel())
    fused_add_mul_activation_kernel[grid](
        in_out_tensor,
        bias,
        in_tensor,
        bias.numel(),
        in_out_tensor.numel(),
        multiplier=0.5,
        activation="relu",
        BLOCK_SIZE=BLOCK_SIZE,
    )
    result_relu = in_out_tensor.clone()

    results = {
        "test_case_1": result_sigmoid[:10].cpu().numpy(),
        "test_case_2": result_relu[:10].cpu().numpy(),
    }
    return results


result_gold = test_fused_add_mul_activation()
