import torch
import triton
import triton.language as tl

AUTOTUNE_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 32,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
        },
        num_stages=4,
        num_warps=8,
    ),
]


def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
    assert w.dim() == 2
    w = w.transpose(0, 1).contiguous()
    assert q_group_size > 1
    assert w.shape[-1] % q_group_size == 0

    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert torch.isnan(scales).sum() == 0

    zeros = min_val + scales * (2 ** (n_bit - 1))
    assert torch.isnan(zeros).sum() == 0

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    assert torch.isnan(out).sum() == 0

    out = out.to(dtype=torch.int32).reshape(w.shape)
    out_uint8 = (out[::, ::2] << 4 | out[::, 1::2]).to(torch.uint8)

    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1)
    scales_and_zeros = (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

    return out_uint8, scales_and_zeros


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    tl.device_assert(K % BLOCK_SIZE_K == 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // 2)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_ak[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs)
        tl.static_assert(b.dtype == tl.int8)

        _4_i8 = tl.full((1,), 4, dtype=tl.int8)
        b_lo = (b << _4_i8) >> _4_i8
        b_hi = b >> _4_i8

        b_f16 = (
            tl.join(b_lo.to(tl.bfloat16), b_hi.to(tl.bfloat16))
            .permute(0, 2, 1)
            .reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)
        )

        accumulator += tl.dot(a, b_f16)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk // 2

    c = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert (
        a.shape[1] == b.shape[0] * 2
    ), f"Incompatible dimensions: {a.shape[1], b.shape[0] * 2}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def pack_2xint4(t):

    t = t.to(torch.int8).reshape(t.shape[0] // 2, 2, t.shape[1]).permute(1, 0, 2)
    return (t[0] & 0xF) | (t[1] << 4)
