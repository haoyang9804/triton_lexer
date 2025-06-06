from __future__ import annotations

import torch
import triton
import numpy as np

from equitriton.sph_harm import triton_kernels as tk


__all__ = [
    "FirstOrderSphericalHarmonics",
    "SecondOrderSphericalHarmonics",
    "ThirdOrderSphericalHarmonics",
    "FourthOrderSphericalHarmonics",
]


def _num_projections(l: int) -> int:

    return 2 * l + 1


def total_projections(l_max: int) -> int:

    return sum([_num_projections(m) for m in range(l_max + 1)])


def make_output_tensor(x: torch.Tensor, l_max: int) -> list[torch.Tensor]:

    total_num_projections = total_projections(l_max)
    last_dim = x.size(-1)
    remainder = x.shape[:-1]

    output = [
        torch.empty((*remainder, last_dim, 1), dtype=x.dtype, device=x.device)
        for _ in range(total_num_projections)
    ]
    return output


def split_tensor_by_l(
    joint_tensor: torch.Tensor, l_max: int, dim: int = -1
) -> list[torch.Tensor]:

    num_projections = [total_projections(l_value) for l_value in range(l_max + 1)]
    proj_indices = list(np.cumsum(num_projections) - 1)

    return torch.tensor_split(joint_tensor, proj_indices, dim=dim)[1:]


def slice_and_dice_tensor(joint_tensor: torch.Tensor) -> list[torch.Tensor]:

    num_slices = joint_tensor.size(-1)
    slice_indices = np.arange(num_slices).tolist()

    result = torch.tensor_split(joint_tensor, slice_indices, dim=-1)[1:]
    return result


class FirstOrderSphericalHarmonics(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        output_tensors = make_output_tensor(x, 1)
        output_tensors[0][:] = 1.0
        block_size = 256
        vector_length = x.numel()

        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_first_order_fwd[num_blocks,](
            x,
            y,
            z,
            *output_tensors[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        ctx.save_for_backward(x, y, z, mask)

        output = torch.cat(output_tensors, dim=-1)

        if isinstance(mask, torch.Tensor):
            output = output[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):

        d_sph_0, d_sph_1_x, d_sph_1_y, d_sph_1_z = slice_and_dice_tensor(grad_output)
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 3:
            x, y, z = saved_tensors
            mask = None
        else:
            x, y, z, mask = saved_tensors

        sqrt3 = 3**0.5

        x_grad = d_sph_1_x * sqrt3
        y_grad = d_sph_1_y * sqrt3
        z_grad = d_sph_1_z * sqrt3

        return x_grad.squeeze(), y_grad.squeeze(), z_grad.squeeze(), mask


class SecondOrderSphericalHarmonics(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        output_tensors = make_output_tensor(x, 2)
        output_tensors[0][:] = 1.0
        block_size = 256
        vector_length = x.numel()

        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_second_order_fwd[num_blocks,](
            x,
            y,
            z,
            *output_tensors[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        ctx.save_for_backward(x, y, z)
        output = torch.cat(output_tensors, dim=-1)

        if isinstance(mask, torch.Tensor):
            output = output[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):

        gradient_collection = slice_and_dice_tensor(grad_output)
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 3:
            x, y, z = saved_tensors
            mask = None
        else:
            x, y, z, mask = saved_tensors
        x_grad = torch.zeros_like(x)
        y_grad = torch.zeros_like(y)
        z_grad = torch.zeros_like(z)
        block_size = 256
        vector_length = x.numel()

        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_second_order_bwd[num_blocks,](
            x,
            y,
            z,
            x_grad,
            y_grad,
            z_grad,
            *gradient_collection[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        return x_grad.squeeze(), y_grad.squeeze(), z_grad.squeeze(), mask


class ThirdOrderSphericalHarmonics(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        output_tensors = make_output_tensor(x, 3)
        output_tensors[0][:] = 1.0
        block_size = 256
        vector_length = x.numel()

        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_third_order_fwd[num_blocks,](
            x,
            y,
            z,
            *output_tensors[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        ctx.save_for_backward(x, y, z, mask)
        output = torch.cat(output_tensors, dim=-1)

        if isinstance(mask, torch.Tensor):
            output = output[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):

        gradient_collection = slice_and_dice_tensor(grad_output)
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 3:
            x, y, z = saved_tensors
            mask = None
        else:
            x, y, z, mask = saved_tensors
        x_grad = torch.zeros_like(x)
        y_grad = torch.zeros_like(y)
        z_grad = torch.zeros_like(z)
        block_size = 256
        vector_length = x.numel()

        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_third_order_bwd[num_blocks,](
            x,
            y,
            z,
            x_grad,
            y_grad,
            z_grad,
            *gradient_collection[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        return x_grad.squeeze(), y_grad.squeeze(), z_grad.squeeze(), mask


class FourthOrderSphericalHarmonics(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        output_tensors = make_output_tensor(x, 4)
        output_tensors[0][:] = 1.0
        block_size = 256
        vector_length = x.numel()

        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_fourth_order_fwd[num_blocks,](
            x,
            y,
            z,
            *output_tensors[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        ctx.save_for_backward(x, y, z, mask)
        output = torch.cat(output_tensors, dim=-1)

        if isinstance(mask, torch.Tensor):
            output = output[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):

        gradient_collection = slice_and_dice_tensor(grad_output)
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 3:
            x, y, z = saved_tensors
            mask = None
        else:
            x, y, z, mask = saved_tensors
        x_grad = torch.zeros_like(x)
        y_grad = torch.zeros_like(y)
        z_grad = torch.zeros_like(z)
        block_size = 256
        vector_length = x.numel()

        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_fourth_order_bwd[num_blocks,](
            x,
            y,
            z,
            x_grad,
            y_grad,
            z_grad,
            *gradient_collection[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        return x_grad.squeeze(), y_grad.squeeze(), z_grad.squeeze(), mask
