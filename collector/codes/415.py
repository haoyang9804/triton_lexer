from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from packaging import version

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit(do_not_specialize=["T"])
def fused_chunk_retention_fwd_kernel(
    q,
    k,
    v,
    o,
    h0,
    ht,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)

    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))

    d_b, d_o, d_h = (
        tl.math.exp2(BT * b_b),
        tl.math.exp2((o_i + 1) * b_b),
        tl.math.exp2((BT - o_i - 1) * b_b),
    )

    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    p_q = tl.make_block_ptr(
        q + i_bh * T * K, (T, K), (K, 1), (0, i_k * BK), (BT, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k + i_bh * T * K, (K, T), (1, K), (i_k * BK, 0), (BK, BT), (0, 1)
    )
    p_v = tl.make_block_ptr(
        v + i_bh * T * V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o + (i_k * B * H + i_bh).to(tl.int64) * T * V,
        (T, V),
        (V, 1),
        (0, i_v * BV),
        (BT, BV),
        (1, 0),
    )

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(
            h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    NT = tl.cdiv(T, BT)
    for i in range(0, NT):

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s

        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False) * d_o[:, None]
            b_h = d_b * b_h + tl.dot(
                b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False
            )
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False) * d_o[:, None]
            if i == NT - 1 and (T % BT) != 0:
                d_b = tl.math.exp2((T % BT) * b_b)
                d_h = tl.math.exp2(((T % BT) - o_i - 1) * b_b)
            b_h = d_b * b_h + tl.dot(
                b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False
            )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def fused_chunk_retention_bwd_kernel(
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    h0,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b) * scale, tl.math.exp2((BT - o_i - 1) * b_b)
    d_b = tl.math.exp2(BT * b_b)

    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(
            h0 + i_bh * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)
        )
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(
            k + i_bh * T * K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_v = tl.make_block_ptr(
            v + i_bh * T * V, (V, T), (1, V), (i_v * BV, i * BT), (BV, BT), (0, 1)
        )
        p_do = tl.make_block_ptr(
            do + i_bh * T * V, (T, V), (V, 1), (i * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_dq = tl.make_block_ptr(
            dq + (i_bh + i_v * B * H).to(tl.int64) * T * K,
            (T, K),
            (K, 1),
            (i * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = (b_do * d_q[:, None]).to(b_do.dtype)

        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)

        b_dq = tl.dot(b_ds, b_k, allow_tf32=False)

        if CHECK and i == 0:
            b_dq += tl.dot(b_dd, b_h.to(b_k.dtype), allow_tf32=False)
            b_h = d_b * b_h + tl.dot(
                (b_v * d_k[None, :]).to(b_k.dtype), b_k, allow_tf32=False
            )
        else:
            b_dq += tl.dot(b_dd, b_h.to(b_k.dtype), allow_tf32=False)
            b_h = d_b * b_h + tl.dot(
                (b_v * d_k[None, :]).to(b_k.dtype), b_k, allow_tf32=False
            )

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    b_h = None
    tl.debug_barrier()
    d_s = tl.trans(d_s)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(
            q + i_bh * T * K, (K, T), (1, K), (i_k * BK, T - i * BT), (BK, BT), (0, 1)
        )
        p_k = tl.make_block_ptr(
            k + i_bh * T * K, (T, K), (K, 1), (T - i * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_v = tl.make_block_ptr(
            v + i_bh * T * V, (T, V), (V, 1), (T - i * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_do = tl.make_block_ptr(
            do + i_bh * T * V, (T, V), (V, 1), (T - i * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_dk = tl.make_block_ptr(
            dk + (i_bh + i_v * B * H).to(tl.int64) * T * K,
            (T, K),
            (K, 1),
            (T - i * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_dv = tl.make_block_ptr(
            dv + (i_bh + i_k * B * H).to(tl.int64) * T * V,
            (T, V),
            (V, 1),
            (T - i * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )

        b_q = tl.load(p_q, boundary_check=(0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = (b_do * d_q[:, None]).to(b_do.dtype)

        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)

        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s

        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)

        b_dv = tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        if CHECK and i == 1:
            b_dk += (
                tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
                * d_k[:, None]
            )
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False) * d_k[:, None]
            b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)
        else:
            b_dk += (
                tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
                * d_k[:, None]
            )
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False) * d_k[:, None]
            b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, initial_state, output_final_state):
        B, H, T, K, V = *k.shape, v.shape[-1]

        BT = 64
        BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4

        o = q.new_empty(NK, B, H, T, V)

        if output_final_state:
            final_state = q.new_empty(
                B, H, K, V, dtype=torch.float, requires_grad=False
            )
        else:
            final_state = None

        CHECK = True
        if version.parse(triton.__version__) < version.parse("2.2.0"):
            import warnings

            warnings.warn(
                "Triton<2.2.0 detected for running this kernel, "
                "which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) "
                "that lead to significant precision loss. "
                "We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. "
                "For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
            )
            CHECK = True

        grid = (NV, NK, B * H)
        fused_chunk_retention_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            initial_state,
            final_state,
            scale,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            CHECK=CHECK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.CHECK = CHECK
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, H, T, K, V = *k.shape, v.shape[-1]
        scale = K**-0.5

        BT = 64
        BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        grid = (NV, NK, B * H)

        fused_chunk_retention_bwd_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            initial_state,
            scale,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            CHECK=ctx.CHECK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def fused_chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    o, final_state = FusedChunkRetentionFunction.apply(
        q, k, v, scale, initial_state, output_final_state
    )
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
