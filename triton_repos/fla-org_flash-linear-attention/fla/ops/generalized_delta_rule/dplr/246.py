from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
    use_cuda_graph,
)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BK"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_dplr_delta_rule_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    gk,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_a = a + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_b = b + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_gk = gk + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_o = o + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[None, :] & mask_v[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[None, :] * V + o_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        tmp = tl.sum(b_h * b_a[None, :], axis=1)
        b_h = exp(b_gk)[None, :] * b_h + (
            tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None]
        )
        b_o = tl.sum(b_h * b_q[None, :], axis=1)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * H * K
        p_k += (-1 if REVERSE else 1) * H * K
        p_a += (-1 if REVERSE else 1) * H * K
        p_b += (-1 if REVERSE else 1) * H * K
        p_gk += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_o += (-1 if REVERSE else 1) * H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[None, :] * V + o_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_dplr_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)

    h0 = initial_state
    if output_final_state:
        ht = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        ht = None
    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    fused_recurrent_dplr_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        gk,
        o,
        h0,
        ht,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        REVERSE=reverse,
    )
    return o, ht


class FusedRecurrentDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: Optional[float] = 1.0,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        o, ht = fused_recurrent_dplr_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        return o, ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass for fused_recurrent_dplr_delta_rule is not implemented and will not be supported. "
            "This kernel is only for inference. "
            "For training, please use `chunk_dplr_delta_rule`."
        )


def fused_recurrent_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    o, final_state = FusedRecurrentDPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        gk,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
    return o, final_state
