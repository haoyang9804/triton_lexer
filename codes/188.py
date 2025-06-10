import triton
import triton.language as tl
import torch
from transformers.models.llama.modeling_llama import logger

from triton.language.extra import libdevice

triton_tanh = libdevice.tanh
triton_cast = tl.cast
MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n: int) -> (
    int,
    int,
):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING,
    SOFTCAP,
    DO_LOGIT_SCALING,
    LOGIT_SCALE,
):

    row_idx = tl.program_id(0)

    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)

    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)

    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )

    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits

    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    c = tl.max(logits, 0)

    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:

        x = tl.load(logits_ptr + label_idx).to(tl.float32)

        if DO_LOGIT_SCALING:
            x = LOGIT_SCALE * x
        if DO_SOFTCAPPING:
            x = SOFTCAP * triton_tanh(x / SOFTCAP)

        loss = logsumexp - x
    else:

        loss = 0.0

    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


@triton.jit
def _cross_entropy_backward(
    logits_ptr,
    logits_row_stride,
    dloss_ptr,
    dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING,
    SOFTCAP,
    DO_LOGIT_SCALING,
    LOGIT_SCALE,
):

    row_idx = tl.program_id(0)

    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    dloss_ptr += row_idx * dloss_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    tanh_term = x
    if DO_SOFTCAPPING:

        tanh_term = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * tanh_term

    logsumexp = tl.load(logsumexp_ptr + row_idx)

    y = tl.exp(x - logsumexp)

    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,
        y,
    )

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    if DO_SOFTCAPPING:
        y = y * (1.0 - tanh_term * tanh_term)

    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)


class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, logits, labels, logit_softcapping: float = 0, logit_scaling: float = 0
    ):
        n_rows: int
        vocab_size: int
        n_rows, vocab_size = logits.shape

        losses = torch.empty(n_rows, dtype=torch.float32, device="cuda")

        DO_SOFTCAPPING: bool = bool(logit_softcapping != 0)
        DO_LOGIT_SCALING: bool = bool(logit_scaling != 0)

        BLOCK_SIZE: int
        num_warps: int

        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
        logsumexp = torch.empty(n_rows, dtype=torch.float32, device="cuda")

        _cross_entropy_forward[(n_rows,)](
            logits,
            logits.stride(0),
            losses,
            logsumexp,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=DO_SOFTCAPPING,
            SOFTCAP=logit_softcapping,
            DO_LOGIT_SCALING=DO_LOGIT_SCALING,
            LOGIT_SCALE=logit_scaling,
            num_warps=num_warps,
        )

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.DO_SOFTCAPPING = DO_SOFTCAPPING
        ctx.logit_softcapping = logit_softcapping
        ctx.DO_LOGIT_SCALING = DO_LOGIT_SCALING
        ctx.logit_scaling = logit_scaling
        return losses

    pass

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows: int
        vocab_size: int
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)

        _cross_entropy_backward[(n_rows,)](
            logits,
            logits.stride(0),
            dlosses,
            dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=ctx.DO_SOFTCAPPING,
            SOFTCAP=ctx.logit_softcapping,
            DO_LOGIT_SCALING=ctx.DO_LOGIT_SCALING,
            LOGIT_SCALE=ctx.logit_scaling,
            num_warps=num_warps,
        )
        return logits, None, None, None


def fast_cross_entropy_loss(
    logits, labels, logit_softcapping=0, logit_scaling=0, n_items=None
):

    batch, seq_len, d = logits.shape
    assert labels.shape == (batch, seq_len)

    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch * seq_len, d),
        labels.view(-1),
        logit_softcapping,
        logit_scaling,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items


def reference_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0):

    if logit_scaling != 0:
        logits = logits * logit_scaling

    if logit_softcapping != 0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    label_mask = labels != -100
    labels_masked = labels.clone()
    labels_masked[~label_mask] = 0

    label_log_probs = log_probs.gather(
        dim=-1, index=labels_masked.unsqueeze(-1)
    ).squeeze(-1)

    label_log_probs = label_log_probs * label_mask

    loss = -label_log_probs.sum() / label_mask.sum()
    return loss


def test_cross_entropy():

    print("Testing Fast Cross Entropy implementation...")

    test_configs = [
        {"name": "Standard", "softcap": 0, "scaling": 0},
        {"name": "With Softcapping", "softcap": 10.0, "scaling": 0},
        {"name": "With Scaling", "softcap": 0, "scaling": 2.0},
        {"name": "With Both", "softcap": 10.0, "scaling": 2.0},
    ]

    for config in test_configs:
        print(f"\nTesting {config['name']} configuration...")

        batch_size, seq_len, vocab_size = 2, 10, 32000
        logits = torch.randn(
            batch_size, seq_len, vocab_size, device="cuda", requires_grad=True
        )

        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        labels[0, 0] = -100

        logits_ref = logits.clone().detach().requires_grad_(True)

        our_loss = fast_cross_entropy_loss(
            logits,
            labels,
            logit_softcapping=config["softcap"],
            logit_scaling=config["scaling"],
        )

        ref_loss = reference_cross_entropy_loss(
            logits_ref,
            labels,
            logit_softcapping=config["softcap"],
            logit_scaling=config["scaling"],
        )

        forward_diff = torch.abs(our_loss - ref_loss).item()
        print(f"Forward pass difference: {forward_diff:.6f}")
        assert (
            forward_diff < 1e-4
        ), f"Forward pass failed for {config['name']} configuration!"

        our_loss.backward()
        ref_loss.backward()

        grad_diff = torch.max(torch.abs(logits.grad - logits_ref.grad)).item()
        print(f"Max gradient difference: {grad_diff:.6f}")
        assert (
            grad_diff < 1e-4
        ), f"Backward pass failed for {config['name']} configuration!"

        logits.grad.zero_()
        logits_ref.grad.zero_()

    print("\nAll tests passed successfully!")
    return True


if __name__ == "__main__":
    test_cross_entropy()
