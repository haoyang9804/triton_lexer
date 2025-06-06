


import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, get_multiprocessor_count, input_guard

sigmoid_fwd_codestring = 
sigmoid_bwd_codestring = 

sigmoid_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(sigmoid_fwd_codestring)
sigmoid_bwd_jit_fn = torch.cuda.jiterator._create_jit_fn(sigmoid_bwd_codestring)


@torch.compiler.disable
def sigmoid_fwd(x):
    return sigmoid_fwd_jit_fn(x)


@torch.compiler.disable
def sigmoid_bwd(x, g):
    return sigmoid_bwd_jit_fn(x, g)


class SigmoidFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return sigmoid_bwd(x, dout)


sigmoid = SigmoidFunction.apply


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_fwd_kernel(
    x,
    y,
    temperature,
    T,
    D: tl.constexpr,
    B: tl.constexpr
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_m = tl.minimum(0., b_x)
    b_z = 1. + exp(-tl.abs(b_x))
    b_y = (b_m - log(b_z)) / temperature
    tl.store(y + o_i, b_y.to(y.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_bwd_kernel(
    x,
    dx,
    dy,
    temperature,
    T,
    D: tl.constexpr,
    B: tl.constexpr
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_dy = tl.load(dy + o_i, mask=m_i, other=0.).to(tl.float32)
    b_dx = b_dy * (1. - tl.sigmoid(b_x)) / temperature
    tl.store(dx + o_i, b_dx.to(dx.dtype.element_ty), mask=m_i)


def logsigmoid_fwd(x: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(T, get_multiprocessor_count(x.device.index)))
    y = torch.empty_like(x)
    logsigmoid_fwd_kernel[(triton.cdiv(T, B),)](
        x=x,
        y=y,
        temperature=temperature,
        T=T,
        D=D,
        B=B
    )
    return y


def logsigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(T, get_multiprocessor_count(x.device.index)))
    dx = torch.empty_like(x)
    logsigmoid_bwd_kernel[(triton.cdiv(T, B),)](
        x=x,
        dx=dx,
        dy=dy,
        temperature=temperature,
        T=T,
        D=D,
        B=B
    )
    return dx


class LogSigmoidFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, temperature):
        ctx.save_for_backward(x,)
        ctx.temperature = temperature
        return logsigmoid_fwd(x, temperature)

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return logsigmoid_bwd(x, dy, ctx.temperature), None


def logsigmoid(x: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    return LogSigmoidFunction.apply(x, temperature)


swish_fwd_codestring = 
swish_bwd_codestring = 

swish_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swish_fwd_codestring)
swish_bwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swish_bwd_codestring)


@torch.compiler.disable
def swish_fwd(x):
    return swish_fwd_jit_fn(x)


@torch.compiler.disable
def swish_bwd(x, g):
    return swish_bwd_jit_fn(x, g)


class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return swish_bwd(x, dout)


swish = SwishFunction.apply









@torch.compile
def bias_gelu(y, bias):
    x = bias + y
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=y.dtype)





@torch.compile
def bias_gelu_bwd(g, y, bias):
    
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    grad_y = ff * g
    return grad_y.to(dtype=y.dtype), grad_y.sum(dim=(0), dtype=bias.dtype)


class GeLUFunction(torch.autograd.Function):

    @staticmethod
    
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply





@torch.compile
def gelu_fwd(x):
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=x.dtype)





@torch.compile
def gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return (ff * g).to(dtype=x.dtype)


class FastGeLUFunction(torch.autograd.Function):
    @staticmethod
    
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, input)
        return tmp


fast_gelu_impl = FastGeLUFunction.apply


@torch.compile
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)


@torch.compile
def sqrelu_fwd(x):
    r = F.relu(x.float())
    return (r * r).to(dtype=x.dtype)


@torch.compile
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x.float())).to(dtype=x.dtype)


class SquaredReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return sqrelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return sqrelu_bwd(grad_output, input)


sqrelu = SquaredReLUFunction.apply


swiglu_fwd_codestring = 
swiglu_bwd_codestring = 

swiglu_fwdbwd_codestring = 


swiglu_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd_jit_fn = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)
swiglu_fwdbwd_jit_fn = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_fwdbwd_codestring, num_outputs=3)


@torch.compiler.disable
def swiglu_fwd(x, y):
    return swiglu_fwd_jit_fn(x, y)


@torch.compiler.disable
def swiglu_bwd(x, y, g):
    return swiglu_bwd_jit_fn(x, y, g)


@torch.compiler.disable
def swiglu_fwdbwd(x, y, g):
    return swiglu_fwdbwd_jit_fn(x, y, g)


@torch.compile
def swiglu_fwd_torch(x, y):
    return (F.silu(x.float()) * y).to(x.dtype)


@torch.compile
def swiglu_bwd_torch(x, y, g):
    dtype = x.dtype
    x, y, g = x.float(), y.float(), g.float()
    x_sigmoid = x.sigmoid()
    x_swish = x * x_sigmoid
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x_swish * g
    return dx.to(dtype), dy.to(dtype)


@torch.compile
def swiglu_fwdbwd_torch(x, y, g):
    dtype = x.dtype
    x, y, g = x.float(), y.float(), g.float()
    x_sigmoid = x.sigmoid()
    x_swish = x * x_sigmoid
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x_swish * g
    z = x_swish * y
    return dx.to(dtype), dy.to(dtype), z.to(dtype)


class SwiGLUFunction(torch.autograd.Function):
    r

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
            return swiglu_fwd_torch(x, y)
        else:
            return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
            return swiglu_bwd_torch(x, y, dout)
        else:
            return swiglu_bwd(x, y, dout)


class SwiGLULinearFunction(torch.autograd.Function):
    r

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        with torch.no_grad():
            if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
                z = swiglu_fwd_torch(x, y)
            else:
                z = swiglu_fwd(x, y)
        out = F.linear(z, weight, bias)
        
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        return out

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        with torch.no_grad():
            if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
                dx, dy, z = swiglu_fwdbwd_torch(x, y, dz)
            else:
                dx, dy, z = swiglu_fwdbwd(x, y, dz)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias


swiglu = SwiGLUFunction.apply


swiglu_linear = SwiGLULinearFunction.apply


ACT2FN = {
    'relu': F.relu,
    'sigmoid': sigmoid,
    'logsigmoid': logsigmoid,
    'silu': swish,
    'swish': swish,
    'sqrelu': sqrelu,
    'gelu': fast_gelu_impl,
    'bias_gelu': bias_gelu_impl,
}
