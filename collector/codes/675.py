import itertools

import torch
import triton
import triton.language as tl


@triton.jit
def triton_sum_kernel_scalar_result(
    input_ptr,
    output_ptr,
    M,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE_M

    offsets = block_start + tl.arange(0, BLOCK_SIZE_M)

    mask = offsets < M

    x = tl.load(input_ptr + offsets, mask=mask, other=mask)

    output = tl.sum(x)

    output_offsets = tl.arange(0, 1)

    tl.store(output_ptr + output_offsets, output)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_NON_REDUCE_DIM": b_nr,
                "BLOCK_SIZE_REDUCE_DIM": b_r,
            },
            num_warps=w,
        )
        for b_nr, b_r, w in itertools.product(
            [2, 4, 8, 16],
            [2, 4, 8, 16],
            [
                2,
                4,
                8,
            ],
        )
    ],
    key=["M", "N"],
)
@triton.jit
def triton_sum_kernel_1D_result_sum_then_buffer(
    input_ptr,
    output_ptr,
    M,
    N,
    BLOCK_SIZE_NON_REDUCE_DIM: tl.constexpr,
    BLOCK_SIZE_REDUCE_DIM: tl.constexpr,
    dim: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    reduce_dim_len = M if dim == 0 else N
    non_reduce_dim_len = N if dim == 0 else M

    buffer = tl.zeros((1, BLOCK_SIZE_NON_REDUCE_DIM), dtype=tl.float32)

    block_start_non_reduce_dim = pid * BLOCK_SIZE_NON_REDUCE_DIM
    offsets_non_reduce_dim = block_start_non_reduce_dim + tl.arange(
        0, BLOCK_SIZE_NON_REDUCE_DIM
    )
    mask_non_reduce_dim = offsets_non_reduce_dim < non_reduce_dim_len

    for block_start_reduce_dim in range(0, reduce_dim_len, BLOCK_SIZE_REDUCE_DIM):
        offsets_reduce_dim = block_start_reduce_dim + tl.arange(
            0, BLOCK_SIZE_REDUCE_DIM
        )
        mask_reduce_dim = offsets_reduce_dim < reduce_dim_len

        idxs, mask = None, None
        if dim == 0:
            idxs = (
                offsets_reduce_dim[:, None] * non_reduce_dim_len
            ) + offsets_non_reduce_dim
            mask = mask_reduce_dim[:, None] & mask_non_reduce_dim
        elif dim == 1:
            idxs = (
                offsets_non_reduce_dim[:, None] * reduce_dim_len
            ) + offsets_reduce_dim
            mask = mask_non_reduce_dim[:, None] & mask_reduce_dim

        input = tl.load(input_ptr + idxs, mask=mask, other=mask)

        buffer += tl.sum(input, axis=dim)

    buffer_view = buffer.reshape(
        (BLOCK_SIZE_NON_REDUCE_DIM,),
    )

    tl.store(output_ptr + offsets_non_reduce_dim, buffer_view, mask=mask_non_reduce_dim)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_NON_REDUCE_DIM": b,
                "BLOCK_SIZE_REDUCE_DIM": b,
            },
            num_warps=w,
        )
        for b, w in itertools.product(
            [2, 4, 8, 16],
            [2, 4, 8],
        )
    ],
    key=["M", "N"],
)
@triton.jit
def triton_sum_kernel_1D_result_buffer_then_sum(
    input_ptr,
    output_ptr,
    M,
    N,
    BLOCK_SIZE_NON_REDUCE_DIM: tl.constexpr,
    BLOCK_SIZE_REDUCE_DIM: tl.constexpr,
    dim: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    reduce_dim_len = M if dim == 0 else N
    non_reduce_dim_len = N if dim == 0 else M

    buffer = tl.zeros(
        (BLOCK_SIZE_REDUCE_DIM, BLOCK_SIZE_NON_REDUCE_DIM), dtype=tl.float32
    )

    block_start_non_reduce_dim = pid * BLOCK_SIZE_NON_REDUCE_DIM
    offsets_non_reduce_dim = block_start_non_reduce_dim + tl.arange(
        0, BLOCK_SIZE_NON_REDUCE_DIM
    )
    mask_non_reduce_dim = offsets_non_reduce_dim < non_reduce_dim_len

    for block_start_reduce_dim in range(0, reduce_dim_len, BLOCK_SIZE_REDUCE_DIM):
        offsets_reduce_dim = block_start_reduce_dim + tl.arange(
            0, BLOCK_SIZE_REDUCE_DIM
        )
        mask_reduce_dim = offsets_reduce_dim < reduce_dim_len

        idxs, mask = None, None
        if dim == 0:
            idxs = (
                offsets_reduce_dim[:, None] * non_reduce_dim_len
            ) + offsets_non_reduce_dim
            mask = mask_reduce_dim[:, None] & mask_non_reduce_dim
        elif dim == 1:
            idxs = (
                offsets_non_reduce_dim[:, None] * reduce_dim_len
            ) + offsets_reduce_dim
            mask = mask_non_reduce_dim[:, None] & mask_reduce_dim

        buffer += tl.load(input_ptr + idxs, mask=mask, other=mask)

    buffer_sum = tl.sum(buffer, axis=dim)

    buffer_view = buffer_sum.reshape(
        (BLOCK_SIZE_NON_REDUCE_DIM,),
    )

    tl.store(output_ptr + offsets_non_reduce_dim, buffer_view, mask=mask_non_reduce_dim)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_K": b},
            num_warps=w,
        )
        for b, w in itertools.product(
            [2, 4, 16, 32, 128, 256],
            [2, 4, 8],
        )
    ],
    key=["N"],
)
@triton.jit
def triton_sum_kernel_2D_result_dim_1(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid % tl.cdiv(K, BLOCK_SIZE_K)

    block_start_n = 0
    block_start_k = pid_k * BLOCK_SIZE_K

    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    mask_n = offsets_n < N
    mask_k = offsets_k < K

    idxs_base = (offsets_n[:, None] * K) + offsets_k
    idxs = idxs_base + (pid_m * N * K)

    mask = mask_n[:, None] & mask_k

    input = tl.load(input_ptr + idxs, mask=mask, other=0)

    output = tl.sum(input, axis=0)

    output_offsets = (pid_m * K) + offsets_k

    tl.store(output_ptr + output_offsets, output, mask=mask_k)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_N": b_n,
                "BLOCK_SIZE_K": b_k,
            },
            num_warps=w,
        )
        for b_n, b_k, w in itertools.product(
            [4**n for n in range(6)], [4**n for n in range(4)], [2, 4, 8]
        )
    ],
    key=["N"],
)
@triton.jit
def triton_sum_kernel_2D_result_dim_1_sum_then_buffer(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid % tl.cdiv(K, BLOCK_SIZE_K)

    buffer = tl.zeros((1, BLOCK_SIZE_K), dtype=tl.float32)

    block_start_k = pid_k * BLOCK_SIZE_K
    offsets_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)
    mask_k = offsets_k < K

    for block_start_n in range(0, N, BLOCK_SIZE_N):
        offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offsets_n < N

        idxs_base = (offsets_n[:, None] * K) + offsets_k
        idxs = idxs_base + (pid_m * N * K)

        mask = mask_n[:, None] & mask_k

        input = tl.load(input_ptr + idxs, mask=mask, other=0)

        buffer += tl.sum(input, axis=0)

    buffer_view = buffer.reshape(
        (BLOCK_SIZE_K,),
    )

    output_offsets = (pid_m * K) + offsets_k

    tl.store(output_ptr + output_offsets, buffer_view, mask=mask_k)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_N": b_n,
                "BLOCK_SIZE_K": b_k,
            },
            num_warps=w,
        )
        for b_n, b_k, w in itertools.product(
            [4**n for n in range(7)], [4**n for n in range(4)], [2, 4, 8]
        )
    ],
    key=["N"],
)
@triton.jit
def triton_sum_kernel_2D_result_dim_1_buffer_then_sum(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid % tl.cdiv(K, BLOCK_SIZE_K)

    buffer = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    block_start_k = pid_k * BLOCK_SIZE_K
    offsets_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)
    mask_k = offsets_k < K

    for block_start_n in range(0, N, BLOCK_SIZE_N):
        offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offsets_n < N

        idxs_base = (offsets_n[:, None] * K) + offsets_k
        idxs = idxs_base + (pid_m * N * K)

        mask = mask_n[:, None] & mask_k

        input = tl.load(input_ptr + idxs, mask=mask, other=0)

        buffer += input

    output = tl.sum(buffer, axis=0)

    output_offsets = (pid_m * K) + offsets_k

    tl.store(output_ptr + output_offsets, output, mask=mask_k)
