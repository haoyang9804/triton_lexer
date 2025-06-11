import math
from typing import Union
import torch
import triton
from triton import language as tl


def heuristics_for_tile_size(max_tile_size, *sizes):
    ndim = len(sizes)
    tile_sizes = [0 for _ in range(ndim)]
    for i in range(ndim):
        size = sizes[ndim - 1 - i]
        tile_size = min(max_tile_size, triton.next_power_of_2(size))
        tile_sizes[ndim - 1 - i] = tile_size
        max_tile_size = max(1, max_tile_size // tile_size)
    return tuple(tile_sizes)


def heuristics_for_num_warps(tile_size):
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


class StridedBuffer:

    def __init__(
        self, base: torch.Tensor, shape=None, strides=None, dtype=None, offset=0
    ):
        self._base = base
        self.dtype = dtype or base.dtype
        if offset == 0:
            self._data_ptr = self._base.data_ptr()
        else:
            offset = self.dtype.itemsize * offset
            self._data_ptr = self._base.data_ptr() + offset
        self.shape = tuple(shape if shape is not None else self._base.shape)
        self._strides = tuple(strides if strides is not None else self._base.stride())
        self.device = self._base.device
        self.ndim = len(self.shape)

    def stride(self):
        return self._strides

    def size(self):
        return self.shape

    def element_size(self):
        return self.dtype.itemsize

    def numel(self):
        return math.prod(self.shape)

    def dim(self):
        return self.ndim

    def unwrap(self):
        return self._base

    def data_ptr(self):
        return self._data_ptr


def relu_forward_wrapper_rank_1(
    in0: Union[torch.Tensor, StridedBuffer],
    /,
    *,
    out0: Union[torch.Tensor, StridedBuffer],
):

    assert in0.shape == out0.shape, "operand shapes mismatch"

    shape = out0.shape
    num_tasks = out0.numel()
    tile_sizes = heuristics_for_tile_size(512, *shape)
    tile_size = math.prod(tile_sizes)
    num_tiles = math.prod(
        triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes)
    )
    num_ctas = min(65536, num_tiles)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = heuristics_for_num_warps(tile_size)
    one_tile_per_cta = tiles_per_cta == 1
    grid = (num_ctas, 1, 1)

    in0_strides = in0.stride()
    in0_stride_order = (0,)
    out0_strides = out0.stride()
    out0_stride_order = (0,)
    with torch.cuda._DeviceGuard(in0.device.index):
        relu_forward_kernel_rank_1[grid](
            in0,
            out0,
            in0_strides[0],
            in0_stride_order[0],
            out0_strides[0],
            out0_stride_order[0],
            shape[0],
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size0=tile_sizes[0],
            one_tile_per_cta=one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0


@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)


@triton.jit
def relu_forward_kernel_rank_1(
    in0_ptr: tl.tensor,
    out0_ptr: tl.tensor,
    in0_stride0: int,
    in0_stride_order0: tl.constexpr,
    out0_stride0: int,
    out0_stride_order0: tl.constexpr,
    s0: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles0 = tl.cdiv(s0, tile_size0)
    if one_tile_per_cta:
        tile_id = pid

        tile_id0 = tile_id

        offset0 = tile_id0 * tile_size0

        in0_bptr = tl.make_block_ptr(
            in0_ptr,
            (s0,),
            (in0_stride0,),
            (offset0,),
            (tile_size0,),
            order=(in0_stride_order0,),
        )
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(
            in0_ptr.type.element_ty
        )

        out0 = relu_forward(in0)

        out0_bptr = tl.make_block_ptr(
            out0_ptr,
            (s0,),
            (out0_stride0,),
            (offset0,),
            (tile_size0,),
            order=(out0_stride_order0,),
        )
        tl.store(
            out0_bptr,
            out0.to(out0_bptr.type.element_ty),
            boundary_check=(out0_stride_order0,),
        )
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas

            tile_id0 = tile_id

            offset0 = tile_id0 * tile_size0

            in0_bptr = tl.make_block_ptr(
                in0_ptr,
                (s0,),
                (in0_stride0,),
                (offset0,),
                (tile_size0,),
                order=(in0_stride_order0,),
            )
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(
                in0_ptr.type.element_ty
            )

            out0 = relu_forward(in0)

            out0_bptr = tl.make_block_ptr(
                out0_ptr,
                (s0,),
                (out0_stride0,),
                (offset0,),
                (tile_size0,),
                order=(out0_stride_order0,),
            )
            tl.store(
                out0_bptr,
                out0.to(out0_bptr.type.element_ty),
                boundary_check=(out0_stride_order0,),
            )


def test_relu_forward():

    device = torch.device("cuda")

    results = {}

    in0 = torch.randn(512, device=device)
    out0 = torch.empty_like(in0)
    relu_forward_wrapper_rank_1(in0, out0=out0)
    results["test_case_1"] = out0

    in0 = torch.randn(100, device=device)
    out0 = torch.empty_like(in0)
    relu_forward_wrapper_rank_1(in0, out0=out0)
    results["test_case_2"] = out0

    in0 = torch.randn(1025, device=device)
    out0 = torch.empty_like(in0)
    relu_forward_wrapper_rank_1(in0, out0=out0)
    results["test_case_3"] = out0

    in0 = torch.randn(4096, device=device)
    out0 = torch.empty_like(in0)
    relu_forward_wrapper_rank_1(in0, out0=out0)
    results["test_case_4"] = out0

    in0 = torch.randn(10000, device=device)
    out0 = torch.empty_like(in0)
    relu_forward_wrapper_rank_1(in0, out0=out0)
    results["test_case_5"] = out0

    base = torch.randn(512, device=device)
    shape = (512,)
    strides = (1,)
    strided_buffer = StridedBuffer(base, shape=shape, strides=strides, dtype=base.dtype)
    out0 = torch.empty_like(base)
    relu_forward_wrapper_rank_1(strided_buffer, out0=out0)
    results["test_case_6"] = out0

    return results


result_gold = test_relu_forward()
