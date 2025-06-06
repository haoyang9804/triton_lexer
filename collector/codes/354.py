from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import input_guard


MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def kl_div_kernel(
    logits,
    target_logits,
    loss,
    s_logits,
    s_loss,
    reduction: tl.constexpr,
    N: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):

    i_n = tl.program_id(0).to(tl.int64)

    logits += i_n * s_logits
    target_logits += i_n * s_logits

    sm = float("-inf")
    tm = float("-inf")

    sd, td = 0.0, 0.0

    NV = tl.cdiv(V, BV)
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)

        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float("-inf"))
        b_sm = tl.max(b_sl)
        m_new = tl.maximum(sm, b_sm)
        sd = sd * exp(sm - m_new) + tl.sum(exp(b_sl - m_new))
        sm = m_new

        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float("-inf"))
        b_tm = tl.max(b_tl)
        m_new = tl.maximum(tm, b_tm)
        td = td * exp(tm - m_new) + tl.sum(exp(b_tl - m_new))
        tm = m_new

    b_loss = 0.0

    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float("-inf"))
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float("-inf"))
        b_sp_log = b_sl - sm - log(sd)
        b_tp_log = b_tl - tm - log(td)
        b_sp = exp(b_sp_log)
        b_tp = exp(b_tp_log)
        b_kl = tl.where(o_x < V, b_tp * (b_tp_log - b_sp_log), 0)
        b_dl = -b_tp + b_sp
        b_loss += tl.sum(b_kl)
        if reduction == "batchmean":
            b_dl = b_dl / N
        tl.store(logits + o_x, b_dl, mask=o_x < V)

    if reduction == "batchmean":
        b_loss = b_loss / N

    tl.store(loss + i_n * s_loss, b_loss)


@triton.jit
def elementwise_mul_kernel(x, g, N: tl.constexpr, B: tl.constexpr):

    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)

    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def fused_kl_div_forward(
    x: torch.Tensor,
    target_x: torch.Tensor,
    weight: torch.Tensor,
    target_weight: torch.Tensor,
    reduction: str = "batchmean",
):
    device = x.device

    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    NC = min(8, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    dx = torch.zeros_like(x, device=device)
    dw = torch.zeros_like(weight, device=device) if weight is not None else None

    loss = torch.zeros(N, dtype=torch.float32, device=device)

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)

        c_sx = x[start:end]
        c_tx = target_x[start:end]

        c_sl = F.linear(c_sx, weight)
        c_tl = F.linear(c_tx, target_weight)

        c_loss = loss[start:end]

        kl_div_kernel[(c_sx.shape[0],)](
            logits=c_sl,
            target_logits=c_tl,
            loss=c_loss,
            s_logits=c_sl.stride(-2),
            s_loss=c_loss.stride(-1),
            reduction=reduction,
            N=N,
            V=V,
            BV=BV,
            num_warps=32,
        )

        dx[start:end] = torch.mm(c_sl, weight)

        if weight is not None:
            torch.addmm(input=dw, mat1=c_sl.t(), mat2=c_sx, out=dw)

    loss = loss.sum()
    return loss, dx, dw


def fused_kl_div_backward(do: torch.Tensor, dx: torch.Tensor, dw: torch.Tensor):

    if torch.ne(do, torch.tensor(1.0, device=do.device)):

        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
            x=dx,
            g=do,
            N=N * H,
            B=B,
            num_warps=32,
        )

        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[(triton.cdiv(V * H, B),)](
                x=dw,
                g=do,
                N=V * H,
                B=B,
                num_warps=32,
            )

    return dx, dw


class FusedKLDivLossFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        target_x: torch.Tensor,
        weight: torch.Tensor,
        target_weight: torch.Tensor,
        reduction: str,
    ):
        loss, dx, dw = fused_kl_div_forward(
            x=x,
            target_x=target_x,
            weight=weight,
            target_weight=target_weight,
            reduction=reduction,
        )
        ctx.save_for_backward(dx, dw)
        return loss

    @staticmethod
    @input_guard
    def backward(ctx, do):
        dx, dw = ctx.saved_tensors
        dx, dw = fused_kl_div_backward(do, dx, dw)
        return dx, None, dw, None, None


def fused_kl_div_loss(
    x: torch.Tensor,
    target_x: torch.Tensor,
    weight: torch.Tensor,
    target_weight: torch.Tensor,
    reduction: str = "batchmean",
) -> Tuple[torch.Tensor, torch.Tensor]:

    return FusedKLDivLossFunction.apply(x, target_x, weight, target_weight, reduction)


class FusedKLDivLoss(nn.Module):

    def __init__(self, reduction: str = "batchmean"):

        super().__init__()

        assert reduction in ["batchmean"], f"reduction: {reduction} is not supported"

        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        target_x: torch.Tensor,
        weight: torch.Tensor,
        target_weight: torch.Tensor,
    ):

        loss = fused_kl_div_loss(
            x=x,
            target_x=target_x,
            weight=weight,
            target_weight=target_weight,
            reduction=self.reduction,
        )
        return loss
