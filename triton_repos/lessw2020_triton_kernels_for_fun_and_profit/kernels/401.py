import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional
from torch.autograd.function import FunctionCtx


@triton.jit
def _fwd_rms_kernel(
    out_ptr_base,
    stride_out_row,
    in_ptr_base,
    stride_x_row,
    stride_x_col,
    weight_ptr,
    rstd,
    num_rows: tl.constexpr,
    num_cols: tl.constexpr,
    block_size: tl.constexpr,
):

    row_index = tl.program_id(0)
    in_ptr_row = in_ptr_base + (row_index * stride_x_row)
    out_ptr_row = out_ptr_base + (row_index * stride_out_row)

    in_block_ptr = tl.make_block_ptr(
        base=in_ptr_base,
        shape=(num_rows, num_cols),
        strides=(stride_x_row, stride_x_col),
        offsets=(row_index, 0),
        block_shape=(1, block_size),
        order=(1, 0),
    )

    variance = 0.0
    eps = 1e-8

    for col_index in range(0, num_cols, block_size):

        col_block = tl.load(in_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        variance += tl.sum(col_block * col_block, axis=None)
        in_block_ptr = tl.advance(in_block_ptr, (0, block_size))

    variance /= num_cols
    rstdev = 1 / tl.sqrt(variance + eps)

    tl.store(rstd + row_index, rstdev)

    for start_col in range(0, num_cols, block_size):
        col_offsets = start_col + tl.arange(0, block_size)

        col_mask = col_offsets < num_cols
        weights = tl.load(weight_ptr + col_offsets, mask=col_mask)

        in_block = tl.load(
            in_ptr_row + col_offsets,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        col_block_rms = in_block * rstdev
        out = weights * col_block_rms

        tl.store(out_ptr_row + col_offsets, out, mask=col_mask)


@triton.jit
def _rms_kernel_bwd_dx(
    dact_ptr,
    dout_ptr,
    in_ptr,
    stride_input_row,
    weight_ptr,
    rstdev,
    nrows,
    ncols,
    block_size_cols: tl.constexpr,
):
    row = tl.program_id(0)
    stride_to_row = row * stride_input_row

    input_row_ptr = in_ptr + stride_to_row
    dact_row_ptr = dact_ptr + stride_to_row
    dout_row_ptr = dout_ptr + stride_to_row

    rstdev = tl.load(rstdev + row)

    for offset in range(0, ncols, block_size_cols):
        cols = offset + tl.arange(0, block_size_cols)
        mask = cols < ncols

        input = tl.load(
            input_row_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dout = tl.load(
            dout_row_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        input_pred = input * rstdev
        wdout = weight * dout
        dact = (wdout - input_pred) * rstdev
        tl.store(dact_ptr, dact, mask=mask)


@triton.jit
def _rms_kernel_bwd_dw(
    input_ptr,
    dout_ptr,
    dweight_ptr,
    rstdev_ptr,
    nrows,
    ncols,
    block_size_row: tl.constexpr,
    block_size_col: tl.constexpr,
):
    row_index = tl.program_id(0)
    cols = row_index * block_size_col + tl.arange(0, block_size_col)
    dw = tl.zeros((block_size_row, block_size_col), dtype=tl.float32)
    unroll: tl.constexpr = 4
    for outer in range(0, nrows, block_size_row * unroll):
        for inner in range(unroll):
            rows = outer + inner * block_size_row + tl.arange(0, block_size_row)
            mask = rows[:, None] < block_size_row & (cols[None, :] < block_size_col)
            offsets = rows[:, None] * block_size_col + cols[None, :]
            input = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            dout = tl.load(dout_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            rstdev = tl.load(rstdev_ptr + rows, mask=rows < block_size_row, other=0.0)
            input_pred = input * rstdev[:, None]
            dw += dout * input_pred
    sum_dw = tl.sum(dw, axis=0)
    tl.store(dweight_ptr, sum_dw, mask=cols < block_size_col)


class TritonRMSNorm2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        weight: Tensor,
    ):

        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])

        nrows, ncols = x.shape

        out = torch.ones_like(x)
        rstdev = torch.empty((nrows,), dtype=torch.float32, device="cuda")

        kb_64 = 65536
        min_block_size = 128
        max_block_size = 4096
        default_num_warps = 8
        max_fused_size = kb_64 // x.element_size()
        block_size = min(max_fused_size, triton.next_power_of_2(ncols))
        block_size = max(block_size, min_block_size)
        block_size = min(block_size, max_block_size)

        base_warps = max(block_size // 256, 1)
        num_warps = min(base_warps, 8)

        grid = (nrows,)
        _fwd_rms_kernel[grid](
            out_ptr_base=out,
            stride_out_row=out.stride(0),
            in_ptr_base=x,
            stride_x_row=x.stride(0),
            stride_x_col=x.stride(1),
            weight_ptr=weight,
            rstd=rstdev,
            num_rows=nrows,
            num_cols=ncols,
            block_size=block_size,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x, weight, rstdev)
        ctx.block_size = block_size
        ctx.num_warps = num_warps

        return out.view(*orig_shape)

    @staticmethod
    def backward(
        ctx,
        dout,
    ):
        assert dout.is_contiguous()

        x, weight, rstdev = ctx.saved_tensors
        dact = torch.empty_like(dout)
        print(f"{dact.shape=}")
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])

        nrows, ncols = x.shape
        print(f"{nrows=}, {ncols=}")

        return None, None
        dweight = torch.empty(
            (weight.shape[0],), dtype=weight.dtype, device=weight.device
        )
        grid = (nrows,)

        _rms_kernel_bwd_dx[grid](
            dact,
            dout,
            x,
            x.stride(0),
            weight,
            rstdev,
            nrows,
            ncols,
            block_size_cols=ctx.block_size,
            num_warps=ctx.num_warps,
        )

        if ncols > 8192:
            block_size_col = 128
            block_size_row = 32
            num_warps = 4
        else:
            block_size_col = 16
            block_size_row = 16
            num_warps = 8

        grid = lambda meta: [triton.cdiv(ncols, meta["block_size_col"])]

        _rms_kernel_bwd_dw[grid](
            x,
            dout,
            dweight,
            rstdev,
            nrows,
            ncols,
            block_size_row=block_size_row,
            block_size_col=block_size_col,
            num_warps=num_warps,
        )

        return dact, dweight


def triton_rmsnorm(
    x: Tensor,
    weight: Tensor,
):
    return TritonRMSNorm2.apply(
        x,
        weight,
    )
