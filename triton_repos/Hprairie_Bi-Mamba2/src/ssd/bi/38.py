from typing import Optional

import math
from packaging import version

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

import triton
import triton.language as tl

from einops import rearrange, repeat


from ssd.bi.ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd
from ssd.bi.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd
from ssd.bi.ssd_chunk_state import _chunk_state_fwd, _chunk_state_bwd_db
from ssd.bi.ssd_chunk_state import _chunk_state_bwd_ddAcs_stable
from ssd.bi.ssd_chunk_state import chunk_state, chunk_state_ref
from ssd.bi.ssd_chunk_state import chunk_state_varlen
from ssd.bi.ssd_state_passing import _state_passing_fwd, _state_passing_bwd
from ssd.bi.ssd_state_passing import state_passing, state_passing_ref
from ssd.bi.ssd_chunk_scan import (
    _chunk_scan_fwd,
    _chunk_scan_bwd_dz,
    _chunk_scan_bwd_dstates,
)
from ssd.bi.ssd_chunk_scan import _chunk_scan_bwd_dC, _chunk_scan_bwd_dcb
from ssd.bi.ssd_chunk_scan import (
    _chunk_scan_bwd_ddAcs_stable_fwd,
    _chunk_scan_bwd_ddAcs_stable_bwd,
)
from ssd.bi.ssd_chunk_scan import chunk_scan, chunk_scan_ref
from ssd.bi.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_prev
from ssd.bi.layernorm_gated import rmsnorm_fn, _layer_norm_fwd, _layer_norm_bwd
from ssd.bi.k_activations import _swiglu_fwd, _swiglu_bwd

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def init_to_zero(names):
    return lambda nargs: [
        nargs[name].zero_() for name in names if nargs[name] is not None
    ]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
            pre_hook=init_to_zero(["ddt_ptr"]),
        ),
    ],
    key=["chunk_size", "hdim", "dstate"],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    x_ptr,
    cb_ptr,
    dout_ptr,
    dt_ptr,
    dA_cumsum_f_ptr,
    dA_cumsum_b_ptr,
    D_ptr,
    b_ptr,
    dstates_f_ptr,
    dstates_b_ptr,
    dx_ptr,
    ddt_ptr,
    dD_ptr,
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_cb_batch,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_dout_batch,
    stride_dout_seqlen,
    stride_dout_head,
    stride_dout_hdim,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_f_batch,
    stride_dA_cs_f_chunk,
    stride_dA_cs_f_head,
    stride_dA_cs_f_csize,
    stride_dA_cs_b_batch,
    stride_dA_cs_b_chunk,
    stride_dA_cs_b_head,
    stride_dA_cs_b_csize,
    stride_D_head,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dstates_f_batch,
    stride_dstates_f_chunk,
    stride_dstates_f_head,
    stride_dstates_f_hdim,
    stride_dstates_f_dstate,
    stride_dstates_b_batch,
    stride_dstates_b_chunk,
    stride_dstates_b_head,
    stride_dstates_b_hdim,
    stride_dstates_b_dstate,
    stride_dx_batch,
    stride_dx_seqlen,
    stride_dx_head,
    stride_dx_hdim,
    stride_ddt_batch,
    stride_ddt_chunk,
    stride_ddt_head,
    stride_ddt_csize,
    stride_dD_batch,
    stride_dD_chunk,
    stride_dD_head,
    stride_dD_csize,
    stride_dD_hdim,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    cb_ptr += (
        pid_b * stride_cb_batch
        + pid_c * stride_cb_chunk
        + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    )
    dout_ptr += (
        pid_b * stride_dout_batch
        + pid_c * chunk_size * stride_dout_seqlen
        + pid_h * stride_dout_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += (
        pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    )
    dA_cumsum_f_ptr += (
        pid_b * stride_dA_cs_f_batch
        + pid_c * stride_dA_cs_f_chunk
        + pid_h * stride_dA_cs_f_head
    )
    dA_cumsum_b_ptr += (
        pid_b * stride_dA_cs_b_batch
        + pid_c * stride_dA_cs_b_chunk
        + pid_h * stride_dA_cs_b_head
    )
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    dstates_f_ptr += (
        pid_b * stride_dstates_f_batch
        + pid_c * stride_dstates_f_chunk
        + pid_h * stride_dstates_f_head
    )
    dstates_b_ptr += (
        pid_b * stride_dstates_b_batch
        + pid_c * stride_dstates_b_chunk
        + pid_h * stride_dstates_b_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dA_cs_f_m = tl.load(
        dA_cumsum_f_ptr + offs_m * stride_dA_cs_f_csize,
        mask=offs_m < chunk_size_limit,
        other=0.0,
    ).to(tl.float32)
    dA_cs_b_m = tl.load(
        dA_cumsum_b_ptr + offs_m * stride_dA_cs_b_csize,
        mask=offs_m < chunk_size_limit,
        other=0.0,
    ).to(tl.float32)

    dA_cs_f_last = tl.load(
        dA_cumsum_f_ptr + (chunk_size - 1) * stride_dA_cs_f_csize
    ).to(tl.float32)
    dA_cs_b_last = tl.load(dA_cumsum_b_ptr).to(tl.float32)
    scale_f = tl.exp(dA_cs_f_last - dA_cs_f_m)
    scale_b = tl.exp(dA_cs_b_last - dA_cs_b_m)

    offs_dstate = tl.arange(
        0,
        (
            BLOCK_SIZE_DSTATE
            if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128
            else BLOCK_SIZE_K
        ),
    )
    b_ptrs = b_ptr + (
        offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate
    )
    dstates_f_ptrs = dstates_f_ptr + (
        offs_n[None, :] * stride_dstates_f_hdim
        + offs_dstate[:, None] * stride_dstates_f_dstate
    )
    dstates_b_ptrs = dstates_b_ptr + (
        offs_n[None, :] * stride_dstates_b_hdim
        + offs_dstate[:, None] * stride_dstates_b_dstate
    )
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(
            b_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate),
            other=0.0,
        )
        dstates_f = tl.load(
            dstates_f_ptrs,
            mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        dstates_f = dstates_f.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates_f) * scale_f[:, None]
        dstates_b = tl.load(
            dstates_b_ptrs,
            mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        dstates_b = dstates_b.to(b_ptr.dtype.element_ty)
        acc += tl.dot(b, dstates_b) * scale_b[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(
                b_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_dstate[None, :] < dstate - k),
                other=0.0,
            )
            dstates_f = tl.load(
                dstates_f_ptrs,
                mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            dstates_f = dstates_f.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates_f) * scale_f[:, None]
            dstates_b = tl.load(
                dstates_b_ptrs,
                mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            dstates_b = dstates_b.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates_b) * scale_b[:, None]
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_f_ptrs += BLOCK_SIZE_K * stride_dstates_f_dstate
            dstates_b_ptrs += BLOCK_SIZE_K * stride_dstates_b_dstate

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (
        offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
    )
    dout_ptrs = dout_ptr + (
        offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim
    )
    dA_cumsum_f_ptrs = dA_cumsum_f_ptr + offs_k * stride_dA_cs_f_csize
    dA_cumsum_b_ptrs = dA_cumsum_b_ptr + offs_k * stride_dA_cs_b_csize
    K_MAX = chunk_size_limit

    K_F_MAX = min((pid_m) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)

        cb = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k),
            other=0.0,
        )
        dout = tl.load(
            dout_ptrs,
            mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )

        if (k <= K_F_MAX + BLOCK_SIZE_M) or (k + BLOCK_SIZE_K >= K_F_MAX):
            dA_cs_f_k = tl.load(
                dA_cumsum_f_ptrs, mask=offs_k < chunk_size - k, other=0.0
            ).to(tl.float32)
            mask_f = (k + offs_k[None, :] >= offs_m[:, None]) & (
                k + offs_k[None, :] < K_MAX
            )
            a_f = tl.where(
                mask_f, tl.exp((dA_cs_f_k[None, :] - dA_cs_f_m[:, None])), 0.0
            )
            dA_cs_b_k = tl.load(
                dA_cumsum_b_ptrs, mask=offs_k < chunk_size - k, other=0.0
            ).to(tl.float32)
            mask_b = k + offs_k[None, :] <= offs_m[:, None]
            cb *= a_f + tl.where(
                mask_b, tl.exp((dA_cs_b_k[None, :] - dA_cs_b_m[:, None])), 0.0
            )

        elif k < K_F_MAX:
            dA_cs_b_k = tl.load(
                dA_cumsum_b_ptrs, mask=offs_k < chunk_size - k, other=0.0
            ).to(tl.float32)
            mask_b = k + offs_k[None, :] <= offs_m[:, None]
            cb *= tl.where(
                mask_b, tl.exp((dA_cs_b_k[None, :] - dA_cs_b_m[:, None])), 0.0
            )
        else:
            dA_cs_f_k = tl.load(
                dA_cumsum_f_ptrs, mask=offs_k < chunk_size - k, other=0.0
            ).to(tl.float32)
            mask_f = (k + offs_k[None, :] >= offs_m[:, None]) & (
                k + offs_k[None, :] < K_MAX
            )
            cb *= tl.where(
                mask_f, tl.exp((dA_cs_f_k[None, :] - dA_cs_f_m[:, None])), 0.0
            )
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_f_ptrs += BLOCK_SIZE_K * stride_dA_cs_f_csize
        dA_cumsum_b_ptrs += BLOCK_SIZE_K * stride_dA_cs_b_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += (
        pid_b * stride_dx_batch
        + pid_c * chunk_size * stride_dx_seqlen
        + pid_h * stride_dx_head
    )
    dx_ptrs = dx_ptr + (
        offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim
    )
    if HAS_D:
        dout_res_ptrs = dout_ptr + (
            offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim
        )
        dout_res = tl.load(
            dout_res_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(
                D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0
            ).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(
        dx_ptrs,
        dx,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
    )

    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    x = tl.load(
        x_ptrs,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
        other=0.0,
    ).to(tl.float32)
    if HAS_D:
        dD_ptr += (
            pid_b * stride_dD_batch
            + pid_c * stride_dD_chunk
            + pid_h * stride_dD_head
            + pid_m * stride_dD_csize
        )
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


def _chunk_scan_chunk_state_bwd_dx(
    x, dt, dA_cumsum_f, dA_cumsum_b, B, CB, dout, dstates_f, dstates_b, D=None, dx=None
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum_f.shape == dt.shape
    assert dA_cumsum_b.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates_f.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dstates_b.shape == (batch, nchunks, nheads, headdim, dstate)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(
            triton.cdiv(chunk_size, BLOCK_SIZE_min),
            batch,
            nchunks,
            nheads,
            headdim if D.dim() == 2 else 1,
            device=D.device,
            dtype=torch.float32,
        )
    else:
        dD = None
    dD_strides = (
        (dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
        if D is not None
        else (0, 0, 0, 0, 0)
    )
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    ddt = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32
    )
    grid_dx = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x,
            CB,
            dout,
            dt,
            dA_cumsum_f,
            dA_cumsum_b,
            D,
            B,
            dstates_f,
            dstates_b,
            dx,
            ddt,
            dD,
            chunk_size,
            headdim,
            dstate,
            batch,
            seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            CB.stride(0),
            CB.stride(1),
            CB.stride(2),
            CB.stride(-1),
            CB.stride(-2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum_f.stride(0),
            dA_cumsum_f.stride(2),
            dA_cumsum_f.stride(1),
            dA_cumsum_f.stride(3),
            dA_cumsum_b.stride(0),
            dA_cumsum_b.stride(2),
            dA_cumsum_b.stride(1),
            dA_cumsum_b.stride(3),
            D.stride(0) if D is not None else 0,
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            dstates_f.stride(0),
            dstates_f.stride(1),
            dstates_f.stride(2),
            dstates_f.stride(3),
            dstates_f.stride(4),
            dstates_b.stride(0),
            dstates_b.stride(1),
            dstates_b.stride(2),
            dstates_b.stride(3),
            dstates_b.stride(4),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dx.stride(3),
            ddt.stride(0),
            ddt.stride(2),
            ddt.stride(1),
            ddt.stride(3),
            dD_strides[1],
            dD_strides[2],
            dD_strides[3],
            dD_strides[0],
            dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22,
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs[
            "BLOCK_SIZE_M"
        ]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return dx, ddt.to(dtype=dt.dtype), dD


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    assert (
        chunk_size >= 32
    ), "Triton Reverse is bugged with < 32 cumsums when running a reverse scan, please use 32 >= chunk length until fixed"
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()

    dA_cumsum_f, dA_cumsum_b, dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit
    )
    states_f, states_b = _chunk_state_fwd(
        B, x, dt, dA_cumsum_f, dA_cumsum_b, states_in_fp32=True
    )
    states_f, final_states_f = _state_passing_fwd(
        rearrange(states_f, "... p n -> ... (p n)"),
        dA_cumsum_f[:, :, :, -1],
        chunk_size=chunk_size,
        out_dtype=C.dtype,
    )
    states_b, final_states_b = _state_passing_fwd(
        rearrange(states_b, "... p n -> ... (p n)"),
        dA_cumsum_b[:, :, :, 0],
        chunk_size=chunk_size,
        out_dtype=C.dtype,
        reverse=True,
    )
    states_f, states_b, final_states_f, final_states_b = [
        rearrange(t, "... (p n) -> ... p n", n=dstate)
        for t in [states_f, states_b, final_states_f, final_states_b]
    ]
    CB = _bmm_chunk_fwd(C, B, chunk_size, output_dtype=torch.float32)
    out, out_x = _chunk_scan_fwd(
        CB, x, dt, dA_cumsum_f, dA_cumsum_b, C, states_f, states_b, D=D, z=z
    )
    return (
        out,
        out_x,
        dt,
        dA_cumsum_f,
        dA_cumsum_b,
        states_f,
        states_b,
        final_states_f,
        final_states_b,
    )


def _mamba_chunk_scan_combined_bwd(
    dout,
    x,
    dt,
    A,
    B,
    C,
    out,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    dfinal_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    dx=None,
    ddt=None,
    dB=None,
    dC=None,
    dz=None,
    recompute_output=False,
):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)

    dt_in = dt.clone()
    dA_cumsum_f, dA_cumsum_b, dt = _chunk_cumsum_fwd(
        dt_in,
        A,
        chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    CB = _bmm_chunk_fwd(C, B, chunk_size, output_dtype=torch.float32)
    states_f, states_b = _chunk_state_fwd(
        B, x, dt, dA_cumsum_f, dA_cumsum_b, states_in_fp32=True
    )
    states_f, _ = _state_passing_fwd(
        rearrange(states_f, "... p n -> ... (p n)"),
        dA_cumsum_f[:, :, :, -1],
        chunk_size=chunk_size,
    )
    states_b, _ = _state_passing_fwd(
        rearrange(states_b, "... p n -> ... (p n)"),
        dA_cumsum_b[:, :, :, 0],
        chunk_size=chunk_size,
        reverse=True,
    )
    states_f = rearrange(states_f, "... (p n) -> ... p n", n=dstate)
    states_b = rearrange(states_b, "... (p n) -> ... p n", n=dstate)
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(
            x,
            z,
            out,
            dout,
            chunk_size=chunk_size,
            has_ddAcs=False,
            D=D,
            dz=dz,
            recompute_output=recompute_output,
        )
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out
    dstates_f, dstates_b = _chunk_scan_bwd_dstates(
        C, dA_cumsum_f, dA_cumsum_b, dout, dtype=states_f.dtype
    )

    dstates_f, ddA_chunk_cumsum_f, states_f = _state_passing_bwd(
        rearrange(states_f, "... p n -> ... (p n)"),
        dA_cumsum_f[:, :, :, -1],
        rearrange(dstates_f, "... p n -> ... (p n)"),
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    dstates_b, ddA_chunk_cumsum_b, states_b = _state_passing_bwd(
        rearrange(states_b, "... p n -> ... (p n)"),
        dA_cumsum_b[:, :, :, 0],
        rearrange(dstates_b, "... p n -> ... (p n)"),
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
        reverse=True,
    )

    states_f = rearrange(states_f, "... (p n) -> ... p n", n=dstate)
    dstates_f = rearrange(dstates_f, "... (p n) -> ... p n", n=dstate)
    states_b = rearrange(states_b, "... (p n) -> ... p n", n=dstate)
    dstates_b = rearrange(dstates_b, "... (p n) -> ... p n", n=dstate)

    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(
        x, dt, dA_cumsum_f, dA_cumsum_b, B, CB, dout, dstates_f, dstates_b, D=D, dx=dx
    )

    dB, ddA_next_f, ddA_next_b = _chunk_state_bwd_db(
        x, dt, dA_cumsum_f, dA_cumsum_b, dstates_f, dstates_b, B=B, ngroups=ngroups
    )

    dC, ddA_cumsum_prev_f, ddA_cumsum_prev_b = _chunk_scan_bwd_dC(
        states_f.to(x.dtype),
        states_b.to(x.dtype),
        dA_cumsum_f,
        dA_cumsum_b,
        dout,
        C=C,
        ngroups=ngroups,
    )

    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum_f, dA_cumsum_b, dout, ngroups=ngroups)

    dCB = dCB.to(CB.dtype)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given)

    if z is None:
        dD = dD_from_x

    ddA_cumsum_prev_f[..., -1] += ddA_chunk_cumsum_f
    ddA_prev_f = ddA_cumsum_prev_f.flip([-1]).cumsum(dim=-1).flip([-1])
    ddA_cumsum_prev_b[..., 0] += ddA_chunk_cumsum_b
    ddA_prev_b = ddA_cumsum_prev_b.cumsum(dim=-1)

    ddA_f = _chunk_scan_bwd_ddAcs_stable_fwd(x, dt, dA_cumsum_f, dout, CB)
    ddA_b = _chunk_scan_bwd_ddAcs_stable_bwd(x, dt, dA_cumsum_b, dout, CB)
    ddA = ddA_b + ddA_f + ddA_next_f + ddA_next_b + ddA_prev_b + ddA_prev_f

    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(
        ddA,
        ddt,
        dt_in,
        A,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        ddt=ddt_given,
    )

    return_vals = (dx, ddt_given, dA, dB_given, dC_given, dD, dz, ddt_bias)
    return return_vals if not recompute_output else (*return_vals, outz)


def selective_scan_bwd(dout, x, dt, A, B, C, D=None, z=None):

    import selective_scan

    batch, seqlen, nheads, headdim = x.shape
    chunk_size = dt.shape[-1]
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    x = rearrange(x, "b l h p -> b (h p) l")
    squeeze_dt = dt.dim() == 4
    if dt.dim() == 4:
        dt = repeat(dt, "b h c l -> b h p c l", p=headdim)
    dt = rearrange(dt, "b h p c l -> b (h p) (c l)", p=headdim)
    squeeze_A = A.dim() == 1
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    B = rearrange(B, "b l g n -> b g n l")
    C = rearrange(C, "b l g n -> b g n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")

    if x.stride(-1) != 1:
        x = x.contiguous()
    if dt.stride(-1) != 1:
        dt = dt.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    _, intermediate, *rest = selective_scan.fwd(
        x, dt.to(dtype=x.dtype), A, B, C, D, z, None, False
    )
    if z is not None:
        out = rest[0]
    else:
        out = None

    dout = rearrange(dout, "b l h p -> b (h p) l")

    if dout.stride(-1) != 1:
        dout = dout.contiguous()

    _, ddt, dA, *rest = selective_scan.bwd(
        x,
        dt.to(dtype=x.dtype),
        A,
        B,
        C,
        D,
        z,
        None,
        dout,
        intermediate,
        out,
        None,
        False,
        False,
    )
    ddt = rearrange(ddt, "b (h p) (c l) -> b h p c l", p=headdim, l=chunk_size)
    if squeeze_dt:
        ddt = ddt.float().sum(dim=2)
    if squeeze_A:
        dA = rearrange(dA, "(h p) n -> h p n", p=headdim).sum(dim=(1, 2))
    return ddt, dA


class MambaChunkScanCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=None,
        z=None,
        dt_bias=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
    ):
        ctx.dt_dtype = dt.dtype
        (
            out,
            out_x,
            dt_out,
            dA_cumsum_f,
            dA_cumsum_b,
            states_f,
            states_b,
            final_states_f,
            final_states_b,
        ) = _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )
        ctx.save_for_backward(
            out if z is None else out_x,
            x,
            dt,
            dA_cumsum_f,
            dA_cumsum_b,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
        )
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        return out if not return_final_states else (out, final_states_f, final_states_b)

    @staticmethod
    def backward(ctx, dout, *args):
        out, x, dt, dA_cumsum_f, dA_cumsum_b, A, B, C, D, z, dt_bias = ctx.saved_tensors
        assert (
            not ctx.return_final_states
        ), "final states are not currently supported in the bwd pass"
        dfinal_states = args[0] if ctx.return_final_states else None
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, *rest = _mamba_chunk_scan_combined_bwd(
            dout,
            x,
            dt,
            A,
            B,
            C,
            out,
            ctx.chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dfinal_states=dfinal_states,
            dt_softplus=ctx.dt_softplus,
            dt_limit=ctx.dt_limit,
        )
        return dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, None, None, None


def bimamba_chunk_scan_combined(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
):

    return MambaChunkScanCombinedFn.apply(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        dt_softplus,
        dt_limit,
        return_final_states,
    )


class MambaSplitConv1dScanCombinedFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size,
        initial_states=None,
        seq_idx=None,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        activation="silu",
        rmsnorm_weight=None,
        rmsnorm_eps=1e-6,
        outproj_weight=None,
        outproj_bias=None,
        headdim=None,
        ngroups=1,
        norm_before_gate=True,
    ):
        assert activation in [None, "silu", "swish"]
        if D.dim() == 1:
            assert headdim is not None
            (nheads,) = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert zxbcdt.shape == (
            batch,
            seqlen,
            2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads,
        )
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, xBC, dt = torch.split(
            zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1
        )
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv = rearrange(
            causal_conv1d_cuda.causal_conv1d_fwd(
                rearrange(xBC, "b s d -> b d s"),
                conv1d_weight,
                conv1d_bias,
                seq_idx,
                None,
                None,
                activation in ["silu", "swish"],
            ),
            "b d s -> b s d",
        )
        x, B, C = torch.split(
            xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        if rmsnorm_weight is None:
            out, out_x, dt_out, dA_cumsum, states, final_states = (
                _mamba_chunk_scan_combined_fwd(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    chunk_size=chunk_size,
                    D=D,
                    z=z,
                    dt_bias=dt_bias,
                    initial_states=initial_states,
                    seq_idx=seq_idx,
                    dt_softplus=True,
                    dt_limit=dt_limit,
                )
            )
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
            if d_nonssm > 0:
                out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
        else:
            out_x, _, dt_out, dA_cumsum, states, final_states = (
                _mamba_chunk_scan_combined_fwd(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    chunk_size=chunk_size,
                    D=D,
                    z=None,
                    dt_bias=dt_bias,
                    initial_states=initial_states,
                    seq_idx=seq_idx,
                    dt_softplus=True,
                    dt_limit=dt_limit,
                )
            )

            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            if d_nonssm == 0:
                out = None
            else:
                out01 = torch.empty(
                    (batch, seqlen, d_nonssm + dim),
                    dtype=x_rms.dtype,
                    device=x_rms.device,
                )
                out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
                _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
            out, _, rstd = _layer_norm_fwd(
                x_rms,
                rmsnorm_weight,
                None,
                rmsnorm_eps,
                z_rms,
                out=out,
                group_size=dim // ngroups,
                norm_before_gate=norm_before_gate,
                is_rms_norm=True,
            )
            if d_nonssm == 0:
                out = rearrange(out, "(b s) d -> b s d", b=batch)
            else:
                out = out01
        ctx.outproj_weight_dtype = (
            outproj_weight.dtype if outproj_weight is not None else None
        )
        if outproj_weight is not None:
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                outproj_bias = (
                    outproj_bias.to(dtype) if outproj_bias is not None else None
                )
            out = F.linear(out, outproj_weight, outproj_bias)
        else:
            assert outproj_bias is None
        ctx.save_for_backward(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            out_x,
            A,
            D,
            dt_bias,
            initial_states,
            seq_idx,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
        )
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        return out if not return_final_states else (out, final_states)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        (
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            out,
            A,
            D,
            dt_bias,
            initial_states,
            seq_idx,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
        ) = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        assert nheads % ctx.ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ctx.ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        recompute_output = outproj_weight is not None
        if recompute_output:
            out_recompute = torch.empty(
                *out.shape[:2], d_nonssm + dim, device=out.device, dtype=out.dtype
            )
            out0_recompute, out1_recompute = out_recompute.split(
                [d_nonssm, dim], dim=-1
            )
        zx0, z, xBC, dt = torch.split(
            zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1
        )

        xBC_conv = rearrange(
            causal_conv1d_cuda.causal_conv1d_fwd(
                rearrange(xBC, "b s d -> b d s"),
                conv1d_weight,
                conv1d_bias,
                seq_idx,
                None,
                None,
                ctx.activation in ["silu", "swish"],
            ),
            "b d s -> b s d",
        )
        x, B, C = torch.split(
            xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)
        dzxbcdt = torch.empty_like(zxbcdt)
        dzx0, dz, dxBC_given, ddt_given = torch.split(
            dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1
        )
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(
            dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1
        )
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
        dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        if outproj_weight is not None:
            dout_og = dout
            dout = F.linear(dout, outproj_weight.t())
        if d_nonssm > 0:
            dout0, dout = dout.split([d_nonssm, dim], dim=-1)
            _swiglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        if rmsnorm_weight is None:
            dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
            dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *rest = (
                _mamba_chunk_scan_combined_bwd(
                    dout,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    out,
                    ctx.chunk_size,
                    D=D,
                    z=z,
                    dt_bias=dt_bias,
                    initial_states=initial_states,
                    dfinal_states=dfinal_states,
                    seq_idx=seq_idx,
                    dt_softplus=True,
                    dt_limit=ctx.dt_limit,
                    dx=dx,
                    ddt=ddt_given,
                    dB=dB,
                    dC=dC,
                    dz=dz,
                    recompute_output=recompute_output,
                )
            )
            out_for_linear = (
                rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
            )
            drmsnorm_weight = None
        else:
            batch = dout.shape[0]
            dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
            dz = rearrange(dz, "b l d -> (b l) d")
            x_rms = rearrange(out, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            out1_recompute = (
                rearrange(out1_recompute, "b s d -> (b s) d")
                if recompute_output
                else None
            )
            dout, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(
                dy_rms,
                x_rms,
                rmsnorm_weight,
                None,
                ctx.rmsnorm_eps,
                None,
                rstd,
                z_rms,
                norm_before_gate=ctx.norm_before_gate,
                is_rms_norm=True,
                recompute_output=recompute_output,
                dz=dz,
                out=out1_recompute if recompute_output else None,
            )
            out_for_linear = out_recompute if recompute_output else None
            dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
            dx, ddt, dA, dB, dC, dD, _, ddt_bias, dinitial_states = (
                _mamba_chunk_scan_combined_bwd(
                    dout,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    out,
                    ctx.chunk_size,
                    D=D,
                    z=None,
                    dt_bias=dt_bias,
                    initial_states=initial_states,
                    dfinal_states=dfinal_states,
                    seq_idx=seq_idx,
                    dt_softplus=True,
                    dt_limit=ctx.dt_limit,
                    dx=dx,
                    ddt=ddt_given,
                    dB=dB,
                    dC=dC,
                )
            )

        if outproj_weight is not None:
            doutproj_weight = torch.einsum("bso,bsd->od", dout_og, out_for_linear)
            doutproj_bias = (
                dout_og.sum(dim=(0, 1)) if outproj_bias is not None else None
            )
        else:
            doutproj_weight, doutproj_bias = None, None
        dxBC_given = rearrange(dxBC_given, "b s d -> b d s")
        dxBC_given, dweight, dbias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            rearrange(xBC, "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
            rearrange(dxBC, "b s d -> b d s"),
            seq_idx,
            None,
            None,
            dxBC_given,
            False,
            ctx.activation in ["silu", "swish"],
        )
        dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        return (
            dzxbcdt,
            dweight,
            dbias,
            ddt_bias,
            dA,
            dD,
            None,
            dinitial_states,
            None,
            None,
            None,
            None,
            drmsnorm_weight,
            None,
            doutproj_weight,
            doutproj_bias,
            None,
            None,
            None,
        )


def mamba_split_conv1d_scan_combined(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    seq_idx=None,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
):

    return MambaSplitConv1dScanCombinedFn.apply(
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size,
        initial_states,
        seq_idx,
        dt_limit,
        return_final_states,
        activation,
        rmsnorm_weight,
        rmsnorm_eps,
        outproj_weight,
        outproj_bias,
        headdim,
        ngroups,
        norm_before_gate,
    )
