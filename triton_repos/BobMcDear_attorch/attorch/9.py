import triton
import triton.language as tl

from .act_kernels import apply_act_func


@triton.jit
def accum_linear(accum, input1, input2, fp16: tl.constexpr, tf32: tl.constexpr):

    if fp16:
        input1 = input1.to(tl.float16)
        input2 = input2.to(tl.float16)

    return accum + tl.dot(input1, input2, allow_tf32=tf32)


@triton.jit
def glu(input1, input2, param, act_func: tl.constexpr):

    return input1 * apply_act_func(input2, None, None, None, param, act_func, False)


@triton.jit
def softmax(input, log: tl.constexpr):

    input = input.to(tl.float32)

    input = input - tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=1)[:, None]

    if log:
        output = input - tl.log(denominator)

    else:
        output = numerator / denominator

    return output


@triton.jit
def calc_mean_and_inv_std(input, last_dim, eps, last_dim_mask: tl.constexpr):

    input = input.to(tl.float32)

    mean = tl.sum(input, axis=1) / last_dim
    diff = tl.where(last_dim_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / last_dim + eps)

    return mean, inv_std


@triton.jit
def update_welford(
    input, prev_count, prev_mean, prev_var, curr_count, mask: tl.constexpr
):

    input = input.to(tl.float32)

    count = prev_count + curr_count
    mean = (tl.sum(input) - curr_count * prev_mean) / count
    deltas = tl.where(mask, (input - mean) * (input - prev_mean), 0.0)
    var = prev_var + tl.sum(deltas)

    return count, mean, var


@triton.jit
def update_ema(prev_ema, new_val, momentum):

    return (1 - momentum) * prev_ema + momentum * new_val


@triton.jit
def standardize(input, mean, inv_std, weight, bias):

    return weight * inv_std * (input - mean) + bias


@triton.jit
def calc_p_loss(
    input, target, param, size, p_loss: tl.constexpr, reduction: tl.constexpr
):

    input = input.to(tl.float32)
    target = target.to(tl.float32)

    diff = input - target

    if p_loss == 0:
        error = tl.where(
            diff < param, 0.5 * diff * diff / param, tl.abs(diff) - 0.5 * param
        )

    elif p_loss == 1:
        error = tl.abs(diff)

    elif p_loss == 2:
        error = diff * diff

    elif p_loss == 3:
        error = tl.where(
            diff < param, 0.5 * diff * diff, param * (tl.abs(diff) - 0.5 * param)
        )

    if reduction == "none":
        output = error

    elif reduction == "mean":
        output = tl.sum(error) / size

    elif reduction == "sum":
        output = tl.sum(error)

    return output


@triton.jit
def nll_loss(input, size, reduction: tl.constexpr):

    input = input.to(tl.float32)

    if reduction == "none":
        output = -input

    elif reduction == "mean":
        output = -tl.sum(input) / size

    elif reduction == "sum":
        output = -tl.sum(input)

    return output


@triton.jit
def cross_entropy_loss(input, pred):

    input = input.to(tl.float32)
    pred = pred.to(tl.float32)

    mx = tl.max(input, axis=1)
    input -= mx[:, None]
    loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx

    return loss
