import triton
import triton.language as tl

from .dropout_kernels import apply_dropout, apply_dropout_grad
from .utils import element_wise_kernel_configs


@triton.jit
def sigmoid(input):

    return 1 / (1 + tl.exp(-input))


@triton.jit
def sigmoid_grad(input):

    output_sigmoid = sigmoid(input)
    return output_sigmoid * (1 - output_sigmoid)


@triton.jit
def logsigmoid(input):

    return tl.log(sigmoid(input))


@triton.jit
def logsigmoid_grad(input):

    return 1 / (1 + tl.exp(input))


@triton.jit
def tanh(input):

    return 2 * sigmoid(2 * input) - 1


@triton.jit
def tanh_grad(input):

    output_tanh = tanh(input)
    return 1 - output_tanh * output_tanh


@triton.jit
def relu(input):

    return tl.maximum(0, input)


@triton.jit
def relu_grad(input):

    return tl.where(input <= 0, 0, 1)


@triton.jit
def gelu(input):

    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input


@triton.jit
def gelu_grad(input):

    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    cdf_grad = 0.39894228 * tl.exp(-0.5 * input * input)
    return cdf_grad * input + cdf


@triton.jit
def silu(input):

    return input * sigmoid(input)


@triton.jit
def silu_grad(input):

    output_sigmoid = sigmoid(input)
    return output_sigmoid * (input * (1 - output_sigmoid) + 1)


@triton.jit
def relu6(input):

    return tl.minimum(relu(input), 6)


@triton.jit
def relu6_grad(input):

    return tl.where((0 < input) & (input < 6), 1, 0)


@triton.jit
def hardsigmoid(input):

    return tl.maximum(0, tl.minimum(1, input / 6 + 0.5))


@triton.jit
def hardsigmoid_grad(input):

    return tl.where((-3 < input) & (input < 3), 1 / 6, 0)


@triton.jit
def hardtanh(input):

    return tl.maximum(-1, tl.minimum(1, input))


@triton.jit
def hardtanh_grad(input):

    return tl.where((-1 < input) & (input < 1), 1, 0)


@triton.jit
def hardswish(input):

    return input * relu6(input + 3) / 6


@triton.jit
def hardswish_grad(input):

    return (relu6(input + 3) + input * relu6_grad(input + 3)) / 6


@triton.jit
def selu(input):

    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * (tl.maximum(0, input) + tl.minimum(0, alpha * (tl.exp(input) - 1)))


@triton.jit
def selu_grad(input):

    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * tl.where(input <= 0, alpha * tl.exp(input), 1)


@triton.jit
def mish(input):

    return input * tanh(tl.log(1 + tl.exp(input)))


@triton.jit
def mish_grad(input):

    exp = tl.exp(input)
    delta = exp * (exp + 2) + 2
    return (
        exp
        * (exp * ((4 * input + 6) + exp * (exp + 4)) + 4 * (input + 1))
        / (delta * delta)
    )


@triton.jit
def softplus(input):

    return tl.log(1 + tl.exp(input))


@triton.jit
def softplus_grad(input):

    return sigmoid(input)


@triton.jit
def softsign(input):

    return input / (1 + tl.abs(input))


@triton.jit
def softsign_grad(input):

    denom = 1 + tl.abs(input)
    return 1 / (denom * denom)


@triton.jit
def tanhshrink(input):

    return input - tanh(input)


@triton.jit
def tanhshrink_grad(input):

    return 1 - tanh_grad(input)


@triton.jit
def leaky_relu(input, negative_slope):

    return relu(input) + negative_slope * tl.minimum(0, input)


@triton.jit
def leaky_relu_grad(input, negative_slope):

    return tl.where(input <= 0, negative_slope, 1)


@triton.jit
def elu(input, alpha):

    return tl.where(input <= 0, alpha * (tl.exp(input) - 1), input)


@triton.jit
def elu_grad(input, alpha):

    return tl.where(input <= 0, alpha * tl.exp(input), 1)


@triton.jit
def celu(input, alpha):

    return relu(input) + tl.minimum(0, alpha * (tl.exp(input / alpha) - 1))


@triton.jit
def celu_grad(input, alpha):

    return tl.where(input <= 0, tl.exp(input / alpha), 1)


@triton.jit
def hardshrink(input, lambd):

    return tl.where(tl.abs(input) < lambd, 0, input)


@triton.jit
def hardshrink_grad(input, lambd):

    return tl.where(tl.abs(input) < lambd, 0, 1)


@triton.jit
def softshrink(input, lambd):

    return tl.where(
        input > lambd, input - lambd, tl.where(input < -lambd, input + lambd, 0)
    )


@triton.jit
def softshrink_grad(input, lambd):

    return tl.where(tl.abs(input) < lambd, 0, 1)


@triton.jit
def apply_act_func(
    input, drop_p, seed, offset, param, act_func: tl.constexpr, dropout: tl.constexpr
):

    if act_func == "sigmoid":
        input = input.to(tl.float32)
        output = sigmoid(input)

    if act_func == "logsigmoid":
        input = input.to(tl.float32)
        output = logsigmoid(input)

    elif act_func == "tanh":
        input = input.to(tl.float32)
        output = tanh(input)

    elif act_func == "relu":
        output = relu(input)

    elif act_func == "gelu":
        input = input.to(tl.float32)
        output = gelu(input)

    elif act_func == "silu":
        input = input.to(tl.float32)
        output = silu(input)

    elif act_func == "relu6":
        output = relu6(input)

    elif act_func == "hardsigmoid":
        output = hardsigmoid(input)

    elif act_func == "hardtanh":
        output = hardtanh(input)

    elif act_func == "hardswish":
        output = hardswish(input)

    elif act_func == "selu":
        input = input.to(tl.float32)
        output = selu(input)

    elif act_func == "mish":
        input = input.to(tl.float32)
        output = mish(input)

    elif act_func == "softplus":
        input = input.to(tl.float32)
        output = softplus(input)

    elif act_func == "softsign":
        output = softsign(input)

    elif act_func == "tanhshrink":
        input = input.to(tl.float32)
        output = tanhshrink(input)

    elif act_func == "leaky_relu":
        output = leaky_relu(input, param)

    elif act_func == "elu":
        input = input.to(tl.float32)
        output = elu(input, param)

    elif act_func == "celu":
        input = input.to(tl.float32)
        output = celu(input, param)

    elif act_func == "hardshrink":
        output = hardshrink(input, param)

    elif act_func == "softshrink":
        output = softshrink(input, param)

    if dropout:
        output = apply_dropout(output, drop_p, seed, offset)

    return output


@triton.jit
def apply_act_func_grad(
    output_grad,
    input,
    drop_p,
    seed,
    offset,
    param,
    act_func: tl.constexpr,
    dropout: tl.constexpr,
):

    if act_func == "sigmoid":
        input = input.to(tl.float32)
        output = sigmoid_grad(input)

    if act_func == "logsigmoid":
        input = input.to(tl.float32)
        output = logsigmoid_grad(input)

    elif act_func == "tanh":
        input = input.to(tl.float32)
        output = tanh_grad(input)

    elif act_func == "relu":
        output = relu_grad(input)

    elif act_func == "gelu":
        input = input.to(tl.float32)
        output = gelu_grad(input)

    elif act_func == "silu":
        input = input.to(tl.float32)
        output = silu_grad(input)

    elif act_func == "relu6":
        output = relu6_grad(input)

    elif act_func == "hardsigmoid":
        output = hardsigmoid_grad(input)

    elif act_func == "hardtanh":
        output = hardtanh_grad(input)

    elif act_func == "hardswish":
        output = hardswish_grad(input)

    elif act_func == "selu":
        input = input.to(tl.float32)
        output = selu_grad(input)

    elif act_func == "mish":
        input = input.to(tl.float32)
        output = mish_grad(input)

    elif act_func == "softplus":
        input = input.to(tl.float32)
        output = softplus_grad(input)

    elif act_func == "softsign":
        output = softsign_grad(input)

    elif act_func == "tanhshrink":
        input = input.to(tl.float32)
        output = tanhshrink_grad(input)

    elif act_func == "leaky_relu":
        output = leaky_relu_grad(input, param)

    elif act_func == "elu":
        input = input.to(tl.float32)
        output = elu_grad(input, param)

    elif act_func == "celu":
        input = input.to(tl.float32)
        output = celu_grad(input, param)

    elif act_func == "hardshrink":
        output = hardshrink_grad(input, param)

    elif act_func == "softshrink":
        output = softshrink_grad(input, param)

    if dropout:
        output_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)

    return output_grad * output


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=["size"],
)
@triton.jit
def act_func_forward_kernel(
    input_pointer,
    output_pointer,
    size,
    drop_p,
    seed,
    param,
    act_func: tl.constexpr,
    dropout: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input = tl.load(input_pointer + offset, mask=mask)
    tl.store(
        output_pointer + offset,
        apply_act_func(input, drop_p, seed, offset, param, act_func, dropout),
        mask=mask,
    )


@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=["size"],
)
@triton.jit
def act_func_backward_kernel(
    output_grad_pointer,
    input_pointer,
    input_grad_pointer,
    size,
    drop_p,
    seed,
    param,
    act_func: tl.constexpr,
    dropout: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input = tl.load(input_pointer + offset, mask=mask)

    tl.store(
        input_grad_pointer + offset,
        apply_act_func_grad(
            output_grad, input, drop_p, seed, offset, param, act_func, dropout
        ),
        mask=mask,
    )
