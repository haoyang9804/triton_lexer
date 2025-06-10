from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .conv_kernels import conv2d_forward_kernel
from .types import Context, Device
from .utils import get_output_dtype


def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> int:

    return (in_size + 2 * padding - kernel_size) // stride + 1


class Conv2dAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride_height: int = 1,
        stride_width: int = 1,
        padding_height: int = 1,
        padding_width: int = 1,
        groups: int = 1,
    ) -> Tensor:

        assert weight.ndim == 4, f"Weights must be 4D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), f"Bias must be 1D, received shape {bias.shape}"

        assert (
            input.shape[1] == groups * weight.shape[1]
        ), f"Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
        assert (
            bias is None or weight.shape[0] == bias.shape[0]
        ), f"Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

        batch_dim, in_feat_dim, in_height, in_width = input.shape
        out_feat_dim, _, kernel_height, kernel_width = weight.shape
        out_height = conv2d_output_size(
            in_height, kernel_height, stride_height, padding_height
        )
        out_width = conv2d_output_size(
            in_width, kernel_width, stride_width, padding_width
        )

        output_dtype = get_output_dtype(input.dtype, autocast="fp16")
        output = torch.empty(
            (batch_dim, out_feat_dim, out_height, out_width),
            device=input.device,
            dtype=output_dtype,
        )

        grid = lambda META: (
            cdiv(
                batch_dim * out_height * out_width,
                META["BLOCK_SIZE_BATCH_HEIGHT_WIDTH"],
            ),
            cdiv(out_feat_dim, META["BLOCK_SIZE_OUT_FEAT"]),
            groups,
        )
        conv2d_forward_kernel[grid](
            input,
            weight,
            output,
            batch_dim,
            in_feat_dim,
            in_height,
            in_width,
            out_feat_dim,
            out_height,
            out_width,
            *input.stride(),
            *weight.stride(),
            *output.stride(),
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            groups=groups,
            fp16=output_dtype is torch.float16,
        )

        if bias is not None:

            output += bias.view(1, -1, 1, 1)

        requires_grad = (
            input.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )

        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.groups = groups
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_dtype = output_dtype
        if requires_grad:
            ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:

        input, weight = ctx.saved_tensors

        input = input.to(ctx.output_dtype)
        weight = weight.to(ctx.output_dtype)

        input_grad = nn.grad.conv2d_input(
            input.shape, weight, output_grad, ctx.stride, ctx.padding, groups=ctx.groups
        )
        weight_grad = nn.grad.conv2d_weight(
            input, weight.shape, output_grad, ctx.stride, ctx.padding, groups=ctx.groups
        )
        bias_grad = (
            output_grad.sum(dim=(0, 2, 3)).to(ctx.output_dtype)
            if ctx.bias_requires_grad
            else None
        )

        return input_grad, weight_grad, bias_grad, None, None, None, None, None


class Conv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        if self.dilation != (1,):
            raise RuntimeError("Convolutional layer only supports dilation of 1.")

        if self.padding_mode != "zeros":
            raise RuntimeError("Convolutional layer only support 'zeros' padding mode.")

    def forward(self, input: Tensor) -> Tensor:
        return Conv2dAutoGrad.apply(
            input.unsqueeze(-1),
            self.weight.unsqueeze(-1),
            self.bias,
            *self.stride,
            1,
            *self.padding,
            0,
            self.groups,
        ).squeeze(-1)


class Conv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        if self.dilation != (1, 1):
            raise RuntimeError(
                "Convolutional layer only supports dilation of 1 and (1, 1)."
            )

        if self.padding_mode != "zeros":
            raise RuntimeError("Convolutional layer only support 'zeros' padding mode.")

    def forward(self, input: Tensor) -> Tensor:
        return Conv2dAutoGrad.apply(
            input, self.weight, self.bias, *self.stride, *self.padding, self.groups
        )
