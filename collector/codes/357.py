import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import input_guard


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": BLOCK_SIZE}, num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )
        for BLOCK_SIZE in [1024, 2048, 4096, 8192]
        for NUM_WARPS in [8, 16, 32]
        for NUM_STAGES in [1, 2, 4]
    ],
    key=["B", "N"],
)
@triton.jit
def grpo_fwd_kernel(
    logits_ptr,
    ref_logp_ptr,
    input_ids_ptr,
    advantages_ptr,
    completion_mask_ptr,
    loss_ptr,
    lse_ptr,
    beta,
    save_kl: tl.constexpr,
    B,
    M,
    N,
    L,
    start_idx,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    off_b = row_idx // L
    N = tl.cast(N, tl.int64)

    loss_ptr += row_idx

    completion_mask_ptr += row_idx
    not_skip = tl.load(completion_mask_ptr).to(tl.int1)
    if not_skip == 1:
        ref_logp_ptr += row_idx
        lse_ptr += row_idx
        advantages_ptr += off_b
        logits_ptr += N * (row_idx + off_b)
        input_ids_ptr += row_idx + (off_b + 1) * start_idx
        base_cols = tl.arange(0, BLOCK_SIZE)

        m_i = -float("inf")
        l_i = 0.0
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(logits_ptr + cols, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_ij = tl.max(logits)
            new_m_i = tl.maximum(m_i, m_ij)
            l_i = l_i * exp(m_i - new_m_i) + tl.sum(exp(logits - new_m_i))
            m_i = new_m_i
        lse = log(l_i) + m_i

        idx = tl.load(input_ids_ptr)
        x = tl.load(logits_ptr + idx).to(tl.float32)
        advantage = tl.load(advantages_ptr).to(tl.float32)
        ref_logp = tl.load(ref_logp_ptr)
        logp = x - lse
        diff = ref_logp - logp
        kl = exp(diff) - diff - 1
        loss = kl * beta - advantage

        tl.store(loss_ptr, loss.to(loss_ptr.dtype.element_ty))
        tl.store(lse_ptr, lse.to(lse_ptr.dtype.element_ty))
        if save_kl:
            tl.store(loss_ptr + M, kl.to(loss_ptr.dtype.element_ty))
    else:

        tl.store(loss_ptr, 0.0)
        if save_kl:
            tl.store(loss_ptr + M, 0.0)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        for NUM_WARPS in [32]
        for NUM_STAGES in [4]
    ],
    key=["B", "N"],
)
@triton.jit
def grpo_bwd_kernel(
    dloss_ptr,
    dlogits_ptr,
    logits_ptr,
    ref_logp_ptr,
    input_ids_ptr,
    advantages_ptr,
    completion_mask_ptr,
    lse_ptr,
    beta,
    B,
    N,
    L,
    start_idx,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    off_b = row_idx // L

    N = tl.cast(N, tl.int64)

    dlogits_ptr += N * (row_idx + off_b)
    base_cols = tl.arange(0, BLOCK_SIZE)
    completion_mask_ptr += row_idx
    not_skip = tl.load(completion_mask_ptr).to(tl.int1)

    if not_skip == 1:
        lse_ptr += row_idx
        dloss_ptr += row_idx
        advantages_ptr += off_b
        ref_logp_ptr += row_idx
        logits_ptr += N * (row_idx + off_b)
        input_ids_ptr += row_idx + (off_b + 1) * start_idx
        dloss = tl.load(dloss_ptr).to(tl.float32)
        lse = tl.load(lse_ptr).to(tl.float32)
        idx = tl.load(input_ids_ptr)
        x = tl.load(logits_ptr + idx).to(tl.float32)
        advantage = tl.load(advantages_ptr).to(tl.float32)
        ref_logp = tl.load(ref_logp_ptr)

        tl.debug_barrier()
        logp = x - lse

        dlogp = (beta * (-1.0 * exp(ref_logp - logp) + 1) - advantage) * dloss

        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(logits_ptr + cols, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            probs = exp(logits - lse)
            dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp

            tl.store(
                dlogits_ptr + cols, dlogits.to(dlogits_ptr.dtype.element_ty), mask=mask
            )
    else:
        dlogits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N

            tl.store(
                dlogits_ptr + cols, dlogits.to(dlogits_ptr.dtype.element_ty), mask=mask
            )


class GrpoLoss(torch.autograd.Function):

    @input_guard
    @staticmethod
    def forward(
        ctx,
        logits,
        ref_logp,
        input_ids,
        advantages,
        beta,
        completion_mask,
        save_kl,
        inplace=True,
    ):
        ctx.input_shape = logits.shape
        B, L_ADD_1, N = ctx.input_shape
        L = L_ADD_1 - 1
        M = B * L
        input_ids_start_index = input_ids.size(1) - L

        if not save_kl:
            loss = torch.empty(B, L, device=logits.device, dtype=torch.float32)
        else:
            loss = torch.empty(B * 2, L, device=logits.device, dtype=torch.float32)

        lse = torch.empty(B, L, device=logits.device, dtype=torch.float32)

        if completion_mask is None:
            completion_mask = torch.ones(B, L, device=logits.device, dtype=torch.int32)
        else:
            loss[:B].masked_fill_(completion_mask.logical_not(), 0.0)

        grpo_fwd_kernel[(M,)](
            logits_ptr=logits,
            ref_logp_ptr=ref_logp,
            input_ids_ptr=input_ids,
            advantages_ptr=advantages,
            completion_mask_ptr=completion_mask,
            loss_ptr=loss,
            lse_ptr=lse,
            beta=beta,
            save_kl=save_kl,
            B=B,
            M=M,
            N=N,
            L=L,
            start_idx=input_ids_start_index,
        )
        ctx.beta = beta
        ctx.save_for_backward(lse, logits, input_ids, advantages, completion_mask)
        ctx.ref_logp = ref_logp
        ctx.inplace = inplace
        return loss

    @input_guard
    @staticmethod
    def backward(ctx, dloss):

        lse, logits, input_ids, advantages, completion_mask = ctx.saved_tensors
        inplace = ctx.inplace
        B, L_ADD_1, N = ctx.input_shape
        L = L_ADD_1 - 1
        M = B * L

        input_ids_start_index = input_ids.size(1) - L

        dlogits = logits if inplace else torch.empty_like(logits)
        BN = min(65536, triton.next_power_of_2(N))

        grpo_bwd_kernel[(M,)](
            dloss_ptr=dloss,
            dlogits_ptr=dlogits,
            logits_ptr=logits,
            ref_logp_ptr=ctx.ref_logp,
            input_ids_ptr=input_ids,
            advantages_ptr=advantages,
            completion_mask_ptr=completion_mask,
            lse_ptr=lse,
            beta=ctx.beta,
            B=B,
            N=N,
            L=L,
            BLOCK_SIZE=BN,
            start_idx=input_ids_start_index,
        )

        dlogits[:, -1, :].fill_(0.0)
        return dlogits.view(*ctx.input_shape), None, None, None, None, None, None, None


def fused_grpo_loss(
    logits,
    ref_logp,
    input_ids,
    advantages,
    beta=0.1,
    completion_mask=None,
    save_kl=False,
    inplace=False,
) -> torch.Tensor:

    out = GrpoLoss.apply(
        logits, ref_logp, input_ids, advantages, beta, completion_mask, save_kl, inplace
    )
    if not save_kl:
        return out
    else:
        return out.chunk(2, axis=0)


def grpo_loss_torch(
    logits,
    ref_logp,
    input_ids,
    advantages,
    beta=0.1,
    completion_mask=None,
    save_kl=False,
):
    def get_log_probs(logits, input_ids):
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids[:, -logits.size(1) :]):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    logits = logits[:, :-1]
    per_token_logps = get_log_probs(logits, input_ids)
    ref_per_token_logps = ref_logp
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps)
        - 1
    )

    per_token_loss = torch.exp(
        per_token_logps - per_token_logps.detach()
    ) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    if completion_mask is not None:
        per_token_loss *= completion_mask
        if save_kl:
            per_token_kl *= completion_mask
    return per_token_loss if not save_kl else (per_token_loss, per_token_kl)


@torch.compile(fullgraph=True)
def grpo_loss_with_old_logps(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    old_logps: torch.Tensor,
    pad_mask: torch.Tensor,
    logits_to_keep: int,
    rewards: torch.Tensor,
    beta: float = 0.2,
    epsilon: float = 0.2,
):

    B = logps.shape[0]
    assert B > 1, "Batch * Num generations should be greater than 1"

    rewards_shaped = rewards.view(-1, B)
    advantages = (rewards_shaped - rewards_shaped.mean(dim=1, keepdim=True)) / (
        rewards_shaped.std(dim=1, keepdim=True) + 1e-8
    )
    advantages = advantages.view(-1)

    per_token_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1

    importance_weights = torch.exp(logps - old_logps)

    importance_weights_clipped = torch.clamp(
        importance_weights, 1 - epsilon, 1 + epsilon
    )

    completion_mask = torch.arange(logits_to_keep, device=logps.device)[None, :] >= 0

    completion_mask = completion_mask & pad_mask

    advantages = advantages.unsqueeze(1)

    token_loss = (
        -(
            torch.min(
                advantages * importance_weights, advantages * importance_weights_clipped
            )
            - beta * per_token_kl
        )
        * completion_mask
    )

    loss = -token_loss.sum() / completion_mask.sum()

    return loss
