import triton
import triton.language as tl
import torch


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _kernel_grouped_gemm_persistent_bf16(
    a_ptr,
    b_ptr,
    c_ptr,
    indices_ptr,
    M_TOTAL: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 128,
    SUPER_GROUP_M: tl.constexpr = 32,
):

    c_type = c_ptr.dtype.element_ty

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):

        tile_m_idx, tile_n_idx = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M
        )

        m_start = tile_m_idx * BLOCK_SIZE_M
        n_start = tile_n_idx * BLOCK_SIZE_N

        if m_start < M_TOTAL:

            offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
            offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):

                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

                mask_m = offs_m < M_TOTAL
                mask_n = offs_n < N
                mask_k = offs_k < K

                mask_a = mask_m[:, None] & mask_k[None, :]
                mask_b = mask_n[:, None] & mask_k[None, :]

                group_idx = m_start // GROUP_SIZE_M
                expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

                a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)

                b_ptrs = (
                    b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)

                accumulator += tl.dot(a, b.T)

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M
            )

            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            mask_m = offs_m < M_TOTAL
            mask_n = offs_n < N
            mask_c = mask_m[:, None] & mask_n[None, :]

            c = accumulator.to(tl.float32)

            c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.store(c_ptrs, c.to(c_type), mask=mask_c)


def _grouped_gemm_persistent(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:

    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    M_total, K = inputs.shape
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    num_experts, N, K_weights = expert_weights.shape

    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"

    output = torch.empty((M_total, N), device=inputs.device, dtype=torch.bfloat16)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (NUM_SMS, 1, 1)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    _kernel_grouped_gemm_persistent_bf16[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=NUM_SMS,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return output


def grouped_gemm_persistent(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    return _grouped_gemm_persistent(inputs, expert_weights, expert_indices)
