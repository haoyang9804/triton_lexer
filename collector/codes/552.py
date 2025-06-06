import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    accum,
    l_i,
    m_i,
    q,
    am,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    block_m: tl.constexpr,
    block_dmodel: tl.constexpr,
    block_n: tl.constexpr,
    stage: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    if stage == 1:
        low = 0
        high = start_m * block_m

    else:
        low = start_m * block_m
        high = (start_m + 1) * block_m
        low = tl.multiple_of(low, block_m)

    K_block_ptr = tl.advance(K_block_ptr, (0, low))
    V_block_ptr = tl.advance(V_block_ptr, (low, 0))

    for start_n in range(low, high, block_n):
        start_n = tl.multiple_of(start_n, block_n)

        k = tl.load(K_block_ptr)
        qk = tl.zeros([block_m, block_n], dtype=tl.float32)

        qk += tl.dot(q, k)

        if stage == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])

            alibi = tl.zeros([block_m, block_n], dtype=tl.float32, ones=False)

            qk = qk * qk_scale

            qk += tl.where(mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        probs = tl.math.exp2(qk)
        l_ij = tl.sum(probs, axis=1)

        delta = tl.math.exp2(m_i - m_ij)
        l_i = l_i * delta + l_ij

        accum = accum * delta[:, None]
        v = tl.load(V_block_ptr)
        accum += tl.dot(probs.to(v.type.element_ty), v)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (block_n, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, block_n))
    return accum, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    amask,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    n_ctx: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_dmodel: tl.constexpr,
    stage: tl.constexpr,
):
    start_m = tl.program_id(0)

    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(n_ctx, block_dmodel),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * block_m, 0),
        block_shape=(block_m, block_dmodel),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(block_dmodel, n_ctx),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(block_dmodel, block_n),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(n_ctx, block_dmodel),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(block_n, block_dmodel),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(n_ctx, block_dmodel),
        strides=(stride_om, stride_on),
        offsets=(start_m * block_m, 0),
        block_shape=(block_m, block_dmodel),
        order=(1, 0),
    )

    offs_m = start_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)

    m_i = tl.zeros([block_m], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_m], dtype=tl.float32) + 1.0
    accum = tl.zeros([block_m, block_dmodel], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q = tl.load(Q_block_ptr)

    am = 500

    if stage & 1:
        accum, l_i, m_i = _attn_fwd_inner(
            accum,
            l_i,
            m_i,
            q,
            am,
            K_block_ptr,
            V_block_ptr,
            start_m,
            qk_scale,
            block_m,
            block_dmodel,
            block_n,
            1,
            offs_m,
            offs_n,
        )
    tl.debug_barrier()

    if stage & 2:
        accum, l_i, m_i = _attn_fwd_inner(
            accum,
            l_i,
            m_i,
            q,
            am,
            K_block_ptr,
            V_block_ptr,
            start_m,
            qk_scale,
            block_m,
            block_dmodel,
            block_n,
            2,
            offs_m,
            offs_n,
        )

    m_i += tl.math.log2(l_i)
    accum = accum / l_i[:, None]
    m_ptrs = M + off_hz * n_ctx + offs_m

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, accum.to(Out.type.element_ty))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, attn_mask=None):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lk in {16, 32, 64, 128}
        assert Lq == Lk and Lk == Lv

        out = torch.empty_like(q)

        block_m = 128
        block_n = 64 if Lk < 65 else 32
        num_stages = 4 if Lk < 65 else 3
        num_warps = 4

        grid_rows = (triton.cdiv(q.shape[2], block_m),)

        grid_cols = (q.shape[0] * q.shape[1], 1)
        grid = grid_rows + grid_cols

        M = torch.empty(
            q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float32
        )
        amask = (
            torch.ones((q.shape[2], q.shape[2]), dtype=q.dtype, device=q.device) * 500
        )

        batch, num_heads, n_ctx, d_head = q.shape

        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            amask,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            batch,
            num_heads,
            n_ctx=n_ctx,
            block_m=block_m,
            block_n=block_n,
            block_dmodel=Lk,
            stage=3,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, out, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return out


attention = _attention.apply
