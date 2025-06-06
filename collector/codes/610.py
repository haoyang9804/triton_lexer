from typing import Optional

import torch
import triton

from liger_kernel.ops.jsd import _jsd_kernel
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from liger_kernel.utils import infer_device


MAX_FUSED_SIZE = 4096 if infer_device() == "xpu" else 65536 // 2


def fused_linear_jsd_forward(
    student_input,
    student_weight,
    teacher_input,
    teacher_weight,
    shift_labels,
    jsd_beta,
    ignore_index,
    has_label,
    temperature,
):
    device = student_input.device
    dtype = student_input.dtype

    BT, H = student_input.shape
    V = student_weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    grad_weight = (
        torch.zeros_like(student_weight, device=device)
        if student_weight.requires_grad
        else None
    )
    grad_input = torch.zeros_like(student_input)

    loss_1d = torch.zeros((BT, V), dtype=torch.float32, device=device)

    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)

        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]

        student_logits_chunk = (student_input_chunk @ student_weight.t()).to(
            torch.float32
        )
        teacher_logits_chunk = (teacher_input_chunk @ teacher_weight.t()).to(
            torch.float32
        )
        chunk_n_rows = student_logits_chunk.shape[0]

        loss_1d_slice = loss_1d[start_idx:end_idx]

        student_logits_chunk = student_logits_chunk / temperature
        teacher_logits_chunk = teacher_logits_chunk / temperature
        student_prob_chunk = torch.log_softmax(student_logits_chunk, dim=-1)
        teacher_prob_chunk = torch.log_softmax(teacher_logits_chunk, dim=-1)

        student_prob_chunk = student_prob_chunk.contiguous()
        teacher_prob_chunk = teacher_prob_chunk.contiguous()

        _jsd_kernel[(chunk_n_rows,)](
            X_ptr=student_prob_chunk,
            X_stride=student_prob_chunk.stride(-2),
            Y_ptr=teacher_prob_chunk,
            Y_stride=teacher_prob_chunk.stride(-2),
            loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-2),
            dX_ptr=student_prob_chunk,
            dX_stride=student_prob_chunk.stride(-2),
            label_ptr=(
                shift_labels[start_idx:end_idx]
                if has_label
                else torch.empty(1, device=device)
            ),
            beta=jsd_beta,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            n_cols=V,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_LABEL=has_label,
        )
        loss_1d[start_idx:end_idx] = loss_1d_slice

        student_logits_chunk = (
            student_prob_chunk
            - torch.softmax(student_logits_chunk, dim=-1)
            * student_prob_chunk.sum(dim=-1, keepdim=True).broadcast_to(
                student_prob_chunk.shape
            )
        ) / temperature

        student_logits_chunk = student_logits_chunk.to(dtype)
        grad_input[start_idx:end_idx] = student_logits_chunk @ student_weight

        if grad_weight is not None:
            grad_weight.add_(student_logits_chunk.t() @ student_input_chunk)

    loss = torch.sum(loss_1d)
    return loss, grad_input, grad_weight


def fused_linear_jsd_backward(grad_output, grad_input, grad_weight):

    if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):

        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

    return grad_input, grad_weight


class LigerFusedLinearJSDFunction(torch.autograd.Function):

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        shift_labels: Optional[torch.Tensor] = None,
        jsd_beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):

        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (
                teacher_input.shape[0],
            ), f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            shift_labels = shift_labels.contiguous()
            has_label = True

        loss, grad_input, grad_weight = fused_linear_jsd_forward(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            shift_labels,
            jsd_beta,
            ignore_index,
            has_label,
            temperature,
        )

        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
        )
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        (grad_input, grad_weight) = ctx.saved_tensors
        grad_input, grad_weight = fused_linear_jsd_backward(
            grad_output, grad_input, grad_weight
        )
        return (grad_input, grad_weight, None, None, None, None, None, None)
