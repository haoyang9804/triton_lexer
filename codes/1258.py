import logging
import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 512}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 16}),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}),
    ],
    key=["M", "N"],
)
@triton.jit
def min_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    min_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("inf"))
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):

        n_offset = start_n + tl.arange(0, BLOCK_N)

        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k

        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset

        inp_vals = tl.load(inp_ptrs, mask=mask, other=float("inf"))
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)

        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M

    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


def min_triton(input_tensor, dim=1):

    assert input_tensor.dim() == 2, "输入张量必须是2D的"

    assert dim == 1, "当前只支持在dim=1上求最小值"

    M, N = input_tensor.shape
    K = 1

    min_values = torch.empty((M, K), dtype=torch.float32, device=input_tensor.device)
    min_indices = torch.empty((M, K), dtype=torch.int64, device=input_tensor.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        K,
    )

    min_kernel[grid](
        input_tensor,
        min_values,
        min_indices,
        M,
        N,
        K,
    )

    return min_values.squeeze(1), min_indices.squeeze(1)


if __name__ == "__main__":
    input_tensor = torch.randn(1024, 2000, device="cuda")
    output_tensor = torch.min(input_tensor, dim=1)
    min_values, min_indices = min_triton(input_tensor, dim=1)

    for i in range(5):
        t1 = time.time()
        output_tensor = torch.min(input_tensor, dim=1)
        torch.cuda.synchronize()
        t2 = time.time()
        print("torch.min:{}".format(t2 - t1))

        t3 = time.time()
        min_values, min_indices = min_triton(input_tensor, dim=1)
        torch.cuda.synchronize()
        t4 = time.time()
        print("triton.min:{}".format(t4 - t3))

    print("-")
