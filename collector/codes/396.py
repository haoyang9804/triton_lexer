from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.fused_recurrent import (
    fused_recurrent_bwd_kernel,
    fused_recurrent_fwd_kernel,
)
from fla.ops.utils import chunk_global_cumsum
from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit
def fused_recurrent_gsa_inference_kernel(
    q,
    k,
    v,
    s,
    g,
    o,
    hk0,
    hv0,
    hkt,
    hvt,
    scale,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG

    b_s = tl.load(s + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = tl.load(g + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = exp(b_g)

    b_ok = tl.zeros([M], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)

        p_hk0 = hk0 + i_bg * K * M + (o_k[None, :]) * M + tl.arange(0, M)[:, None]

        mask_k = o_k < K

        mask_hk = (tl.arange(0, M) < M)[:, None] & mask_k[None, :]

        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.0).to(tl.float32)

        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.0).to(tl.float32) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)

        if i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[:, None]
            tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), mask=mask_hk)

    b_qv = tl.softmax(b_ok)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)

        p_hv0 = hv0 + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]

        mask_v = o_v < V

        mask_hv = mask_v[:, None] & (tl.arange(0, M) < M)[None, :]

        b_hv = tl.load(p_hv0, mask=mask_hv, other=0).to(tl.float32)

        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)

        tl.store(o + i_bh * V + o_v, b_ov.to(o.dtype.element_ty), mask=mask_v)

        if i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
            tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), mask=mask_hv)


def fused_recurrent_gsa_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.0,
) -> torch.Tensor:
    B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    HQ = q.shape[2]
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NG = HQ // H

    if initial_state != (None, None) and initial_state is not None:
        hk0, hv0 = initial_state
    else:
        hk0, hv0 = q.new_zeros(B, H, K, M, dtype=torch.float), q.new_zeros(
            B, H, M, V, dtype=torch.float
        )

    hkt, hvt = None, None
    if output_final_state:
        if NG == 1:
            hkt, hvt = hk0, hv0
        else:
            hkt, hvt = q.new_empty(B, H, K, M, dtype=torch.float), q.new_empty(
                B, H, M, V, dtype=torch.float
            )

    o = v.new_empty(B, T, HQ, V)
    grid = (B * HQ,)
    fused_recurrent_gsa_inference_kernel[grid](
        q,
        k,
        v,
        s,
        g,
        o,
        hk0,
        hv0,
        hkt,
        hvt,
        scale=scale,
        K=K,
        V=V,
        M=M,
        BK=BK,
        BV=BV,
        NG=NG,
    )
    return o, (hkt, hvt)


def fused_recurrent_gsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale: float = 1.0,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    HQ = q.shape[2]
    if HQ != H:
        raise ValueError("GQA not supported yet.")

    BK, BV, BM = (
        min(triton.next_power_of_2(K), 64),
        min(triton.next_power_of_2(V), 64),
        min(M, 64),
    )
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)

    hk0, hv0 = None, None
    if initial_state != (None, None) and initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        hkt, hvt = q.new_empty(N, H, K, M, dtype=torch.float), q.new_empty(
            N, H, M, V, dtype=torch.float
        )

    ok = q.new_empty(NK, *s.shape, dtype=torch.float)
    gk, gv = None, g
    grid = (NM, NK, N * H)
    fused_recurrent_fwd_kernel[grid](
        q=q,
        k=k,
        v=s,
        g=None,
        gk=gk,
        gv=gv,
        o=ok,
        h0=hk0,
        ht=hkt,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_G=False,
        USE_GK=False,
        USE_GV=True,
        REVERSE=reverse,
    )
    ok = ok.sum(0)

    qv = ok.softmax(-1, dtype=torch.float)
    ov = q.new_empty(NM, *v.shape, dtype=torch.float)
    gk, gv = g, None
    grid = (NV, NM, N * H)
    fused_recurrent_fwd_kernel[grid](
        q=qv,
        k=s,
        v=v,
        g=None,
        gk=gk,
        gv=gv,
        o=ov,
        h0=hv0,
        ht=hvt,
        cu_seqlens=cu_seqlens,
        scale=1.0,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_G=False,
        USE_GK=True,
        USE_GV=False,
        REVERSE=reverse,
    )
    ov = ov.sum(0)
    return ok, hkt, qv, ov, hvt


def fused_recurrent_gsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    qv: torch.Tensor,
    hk0: Optional[torch.Tensor] = None,
    hv0: Optional[torch.Tensor] = None,
    ok: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    dhkt: Optional[torch.Tensor] = None,
    dhvt: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor]:
    B, T, H, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV, BM = min(K, 64), min(V, 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)

    dqv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
    dsv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
    dv = q.new_empty(NM, B, T, H, V, dtype=torch.float)
    dhk0 = torch.empty_like(hk0) if hk0 is not None else None
    dhv0 = torch.empty_like(hv0) if hv0 is not None else None

    gk, gv = g, None
    grid = (NV, NM, N * H)
    fused_recurrent_bwd_kernel[grid](
        q=qv,
        k=s,
        v=v,
        g=None,
        gk=gk,
        gv=gv,
        h0=hv0,
        do=do,
        dq=dqv,
        dk=dsv,
        dv=dv,
        dht=dhvt,
        dh0=dhv0,
        cu_seqlens=cu_seqlens,
        scale=1.0,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_G=False,
        USE_GK=True,
        USE_GV=False,
        REVERSE=reverse,
    )
    dqv = dqv.sum(0)
    dsv = dsv.sum(0)
    dv = dv.sum(0)
    dgk = chunk_global_cumsum(
        dqv * qv.float() - dsv * s.float(), reverse=not reverse, cu_seqlens=cu_seqlens
    )

    dok = qv * (dqv - (qv * dqv).sum(-1, True))
    dq = q.new_empty(NM, B, T, H, K, dtype=torch.float)
    dk = q.new_empty(NM, B, T, H, K, dtype=torch.float)
    dsk = q.new_empty(NK, B, T, H, M, dtype=torch.float)
    gk, gv = None, g
    grid = (NM, NK, N * H)
    fused_recurrent_bwd_kernel[grid](
        q=q,
        k=k,
        v=s,
        g=None,
        gk=gk,
        gv=gv,
        h0=hk0,
        do=dok,
        dq=dq,
        dk=dk,
        dv=dsk,
        dht=dhkt,
        dh0=dhk0,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_G=False,
        USE_GK=False,
        USE_GV=True,
        REVERSE=reverse,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dsk = dsk.sum(0)

    dgv = chunk_global_cumsum(
        dok.float() * ok.float() - dsk * s.float(),
        reverse=not reverse,
        cu_seqlens=cu_seqlens,
    )

    ds = dsk.add_(dsv)
    dg = dgk.add_(dgv)

    return dq, dk, dv, ds, dg, dhk0, dhv0


class FusedRecurrentGSAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: Optional[float] = None,
        hk0: Optional[torch.Tensor] = None,
        hv0: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        T = q.shape[1]
        if T == 1 and not q.requires_grad:
            o, (hkt, hvt) = fused_recurrent_gsa_inference(
                q=q,
                k=k,
                v=v,
                s=s,
                g=g,
                initial_state=(hk0, hv0),
                output_final_state=output_final_state,
                scale=scale,
            )
            return o, hkt, hvt
        ok, hkt, qv, ov, hvt = fused_recurrent_gsa_fwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            scale=scale,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, s, g, qv, hk0, hv0, ok)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        return ov.to(q.dtype), hkt, hvt

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dhkt=None, dhvt=None):
        q, k, v, s, g, qv, hk0, hv0, ok = ctx.saved_tensors
        scale = ctx.scale
        reverse = ctx.reverse
        cu_seqlens = ctx.cu_seqlens

        if dhkt is not None or dhvt is not None:
            if g is not None:
                assert (
                    g.requires_grad is False
                ), "Cannot load final state gradient and use gates at the same time"
        dq, dk, dv, ds, dg, dhk0, dhv0 = fused_recurrent_gsa_bwd(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            qv=qv,
            hk0=hk0,
            hv0=hv0,
            ok=ok,
            do=do,
            dhkt=dhkt,
            dhvt=dhvt,
            scale=scale,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            ds.to(s),
            dg.to(g),
            None,
            dhk0,
            dhv0,
            None,
            None,
            None,
        )


def fused_recurrent_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False,
    reverse: Optional[bool] = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if (
            initial_state is not None
            and initial_state[0].shape[0] != len(cu_seqlens) - 1
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if initial_state is None:
        initial_state = (None, None)
    o, *final_state = FusedRecurrentGSAFunction.apply(
        q,
        k,
        v,
        s,
        g,
        scale,
        *initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
    return o, final_state
