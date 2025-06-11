import triton
import triton.language as tl
from triton.language import constexpr as const


def cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def offset_1d(sz: const, n_prev_chunks=0):
    return n_prev_chunks * sz + tl.arange(0, sz)


@triton.jit
def offset_2d(offs0, offs1, stride0, stride1=1):
    return tl.expand_dims(offs0, 1) * stride0 + tl.expand_dims(offs1, 0) * stride1


@triton.jit
def mask_1d(offs, max):
    return offs < max


@triton.jit
def mask_2d(offs0, offs1, max0, max1):
    return (tl.expand_dims(offs0, 1) < max0) & (tl.expand_dims(offs1, 0) < max1)


@triton.jit
def load_1d(ptr, sz: const, n, max, stride=1):

    offs = offset_1d(sz, n)
    mask = mask_1d(offs, max)
    return tl.load(ptr + offs, mask)


@triton.jit
def load_full_1d(ptr, sz: const, stride=1):

    offs = offset_1d(sz)
    mask = mask_1d(offs, sz)
    return tl.load(ptr + offs, mask)


@triton.jit
def load_2d(ptr, sz0: const, sz1: const, n0, n1, max0, max1, stride0=None, stride1=1):

    stride0 = stride0 or sz1
    offs0 = offset_1d(sz0, n0)
    offs1 = offset_1d(sz1, n1)
    offs = offset_2d(offs0, offs1, stride0, stride1)
    mask = mask_2d(offs0, offs1, max0, max1)
    return tl.load(ptr + offs, mask)


@triton.jit
def load_full_2d(ptr, sz0: const, sz1: const, stride0=None, stride1=1):

    stride0 = stride0 or sz1
    offs = offset_2d(tl.arange(0, sz0), tl.arange(0, sz1), stride0, stride1)
    mask = mask_2d(tl.arange(0, sz0), tl.arange(0, sz1), sz0, sz1)
    return tl.load(ptr + offs, mask)


@triton.jit
def store_1d(vals, ptr, sz: const, n, max, stride=1):

    offs = offset_1d(sz, n)
    mask = mask_1d(offs, max)
    tl.store(ptr + offs, vals, mask)


@triton.jit
def store_full_1d(vals, ptr, sz: const, stride=1):

    offs = offset_1d(sz)
    mask = mask_1d(offs, sz)
    tl.store(ptr + offs, vals, mask)


@triton.jit
def store_2d(
    vals, ptr, sz0: const, sz1: const, n0, n1, max0, max1, stride0=None, stride1=1
):

    stride0 = stride0 or sz1
    offs0 = offset_1d(sz0, n0)
    offs1 = offset_1d(sz1, n1)
    offs = offset_2d(offs0, offs1, stride0, stride1)
    mask = mask_2d(offs0, offs1, max0, max1)
    tl.store(ptr + offs, vals, mask)


@triton.jit
def store_full_2d(vals, ptr, sz0: const, sz1: const, stride0=None, stride1=1):

    stride0 = stride0 or sz1
    offs = offset_2d(tl.arange(0, sz0), tl.arange(0, sz1), stride0, stride1)
    mask = mask_2d(tl.arange(0, sz0), tl.arange(0, sz1), sz0, sz1)
    tl.store(ptr + offs, vals, mask)
