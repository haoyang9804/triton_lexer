import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


ROPE_GROUP_SIZE = 4


@triton.heuristics(
    {
        "BACKWARD_PASS": lambda args: args["BACKWARD_PASS"],
    }
)
@triton.jit
def _rope_embedding(
    Q,
    Q_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROPE_GROUP_SIZE: tl.constexpr = 4,
):

    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(
        sin
        + (row_position % seqlen) * sin_row_stride
        + half_head_dim * 0
        + col_offsets,
        mask=mask,
        other=0,
    )
    cos1 = tl.load(
        cos
        + (row_position % seqlen) * cos_row_stride
        + half_head_dim * 0
        + col_offsets,
        mask=mask,
        other=0,
    )

    if BACKWARD_PASS:

        sin1 = -sin1

    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = (
            row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim
        )

        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)


def _rope_embedding_forward_impl(Q, cos, sin):
    Q = Q.transpose(1, 2).clone()
    cos, sin = cos.squeeze(), sin.squeeze()
    batch, seq_len, n_heads, head_dim = Q.shape
    Q = Q.reshape(batch * seq_len, n_heads * head_dim)
    n_rows, n_cols = Q.shape
    assert seq_len <= cos.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)

    div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
    n_groups = div + (mod != 0)

    _rope_embedding[
        (
            n_rows,
            n_groups,
        )
    ](
        Q,
        Q.stride(0),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        seq_len,
        head_dim,
        n_heads,
        BACKWARD_PASS=False,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    Q = Q.view(batch, seq_len, n_heads, head_dim)
    Q = Q.transpose(1, 2)
    return Q, cos, sin, n_groups, BLOCK_SIZE, num_warps


def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps):
    dY = dY.transpose(1, 2)
    batch, seq_len, n_heads, head_dim = dY.shape
    dY = dY.reshape(batch * seq_len, n_heads * head_dim)

    n_rows, n_cols = dY.shape

    _rope_embedding[
        (
            n_rows,
            n_groups,
        )
    ](
        dY,
        dY.stride(0),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        seq_len,
        head_dim,
        n_heads,
        BACKWARD_PASS=True,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dY = dY.view(batch, seq_len, n_heads, head_dim)
    dY = dY.transpose(1, 2)
    return dY


def test_rope_embedding_forward():

    batch, seq_len, n_heads, head_dim = 2, 16, 8, 64
    Q = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, device="cuda")

    Q_out, cos_out, sin_out, n_groups, BLOCK_SIZE, num_warps = (
        _rope_embedding_forward_impl(Q, cos, sin)
    )

    dY = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    dY_out = _rope_embedding_backward_impl(
        dY, cos, sin, n_groups, BLOCK_SIZE, num_warps
    )

    results = {}

    batch, seq_len, n_heads, head_dim = 1, 8, 4, 32
    Q = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, device="cuda")
    Q_out, cos_out, sin_out, n_groups, BLOCK_SIZE, num_warps = (
        _rope_embedding_forward_impl(Q, cos, sin)
    )
    dY = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    dY_out = _rope_embedding_backward_impl(
        dY, cos, sin, n_groups, BLOCK_SIZE, num_warps
    )
    results["test_case_1"] = (Q_out, dY_out)

    batch, seq_len, n_heads, head_dim = 4, 32, 16, 128
    Q = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, device="cuda")
    Q_out, cos_out, sin_out, n_groups, BLOCK_SIZE, num_warps = (
        _rope_embedding_forward_impl(Q, cos, sin)
    )
    dY = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    dY_out = _rope_embedding_backward_impl(
        dY, cos, sin, n_groups, BLOCK_SIZE, num_warps
    )
    results["test_case_2"] = (Q_out, dY_out)

    batch, seq_len, n_heads, head_dim = 8, 64, 32, 256
    Q = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, device="cuda")
    Q_out, cos_out, sin_out, n_groups, BLOCK_SIZE, num_warps = (
        _rope_embedding_forward_impl(Q, cos, sin)
    )
    dY = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    dY_out = _rope_embedding_backward_impl(
        dY, cos, sin, n_groups, BLOCK_SIZE, num_warps
    )
    results["test_case_3"] = (Q_out, dY_out)

    batch, seq_len, n_heads, head_dim = 16, 128, 64, 512
    Q = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, device="cuda")
    Q_out, cos_out, sin_out, n_groups, BLOCK_SIZE, num_warps = (
        _rope_embedding_forward_impl(Q, cos, sin)
    )
    dY = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda")
    dY_out = _rope_embedding_backward_impl(
        dY, cos, sin, n_groups, BLOCK_SIZE, num_warps
    )
    results["test_case_4"] = (Q_out, dY_out)

    return results


result_gold = test_rope_embedding_forward()
