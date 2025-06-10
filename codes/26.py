from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from triton import cdiv

from .cross_entropy_loss_kernels import (
    cross_entropy_loss_backward_kernel,
    cross_entropy_loss_forward_kernel,
)
from .softmax_kernels import BLOCK_SIZE_BATCH_heuristic
from .types import Context
from .utils import get_output_dtype


class CrossEntropyLossAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Context,
        input: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
    ) -> Tensor:

        assert input.ndim == 2, f"Inputs of rank other than 2 not valid"
        assert len(input) == len(
            target
        ), f"Incompatible input shape ({input.shape}) and target shape ({target.shape})"
        assert (
            weight is None or len(weight) == input.shape[1]
        ), f"Dimensionality of weight vector ({len(weight)}) and input features ({input.shape[1]}) not equal"

        batch_dim, feat_dim = input.shape
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic(
            {"batch_dim": batch_dim, "feat_dim": feat_dim}
        )
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
        weighted = weight is not None

        output_dtype = get_output_dtype(input.dtype, autocast="fp32")
        output = torch.empty(out_batch_dim, dtype=output_dtype, device=input.device)

        if weighted:
            sum_weights = torch.empty_like(output, dtype=torch.float32)

        else:
            sum_weights = None

        grid = lambda META: (cdiv(len(input), META["BLOCK_SIZE_BATCH"]),)
        cross_entropy_loss_forward_kernel[grid](
            input,
            target,
            weight,
            sum_weights,
            output,
            batch_dim,
            feat_dim,
            *input.stride(),
            weighted=weighted,
        )
        output = output.sum()

        if weighted:
            sum_weights = sum_weights.sum()
            output /= sum_weights

        ctx.sum_weights = sum_weights
        ctx.weight = weight
        ctx.output_dtype = output_dtype
        if input.requires_grad:
            ctx.save_for_backward(input, target)

        return output

    @staticmethod
    def backward(
        ctx: Context,
        output_grad: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:

        (input, target) = ctx.saved_tensors
        batch_dim, feat_dim = input.shape
        input_grad = torch.empty_like(input, dtype=ctx.output_dtype)

        grid = lambda META: (cdiv(len(input), META["BLOCK_SIZE_BATCH"]),)
        cross_entropy_loss_backward_kernel[grid](
            output_grad,
            target,
            input,
            ctx.weight,
            ctx.sum_weights,
            input_grad,
            batch_dim,
            feat_dim,
            *input.stride(),
            *input_grad.stride(),
            weighted=ctx.weight is not None,
        )

        return input_grad, None, None


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(
        self,
        reduction: str = "mean",
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )

        if self.reduction != "mean":
            raise RuntimeError("Cross entropy only supports averaging the loss.")

        if label_smoothing > 0.0:
            raise RuntimeError("Cross entropy does not support label smoothing.")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return CrossEntropyLossAutoGrad.apply(input, target, self.weight)
