import functools
import math
from typing import Optional


import torch
import torch.nn as nn
import triton
import triton.language as tl


def make_quant(model, bits, groupsize):

    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue

        if name == "lm_head":
            continue

        qlayer = QuantLinear(
            bits, groupsize, m.in_features, m.out_features, m.bias is not None
        )
        parent_name = name.rsplit(".", 1)[0]
        parent = model.get_submodule(parent_name)

        setattr(parent, name[len(parent_name) + 1 :], qlayer)


def autotune_warmup(model):

    modules = (m for m in model.modules() if isinstance(m, QuantLinear))
    kn_values = {
        (m.infeatures, m.outfeatures): (m.qweight, m.scales, m.qzeros, m.groupsize)
        for m in modules
    }

    print(f"QuantLinear Warmup: Found {len(kn_values)} unique KN values.")

    def func(m, k, qweight, scales, qzeros, groupsize):
        a = torch.randn(1, m, k, dtype=torch.float16, device="cuda")
        triton_matmul4(groupsize, a, qweight, scales, qzeros)

    return (
        functools.partial(
            func,
            k=k,
            qweight=qweight,
            scales=scales,
            qzeros=qzeros,
            groupsize=groupsize,
        )
        for (k, n), (qweight, scales, qzeros, groupsize) in kn_values.items()
    )


class QuantLinear(nn.Module):
    def __init__(
        self, bits: int, groupsize: int, infeatures: int, outfeatures: int, bias: bool
    ):
        super().__init__()

        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")

        groupsize = infeatures if groupsize == -1 else groupsize

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize

        features_per_int = 32 // bits

        assert (
            outfeatures % features_per_int == 0
        ), "outfeatures must be a multiple of features_per_int"

        self.register_buffer(
            "qweight",
            torch.empty(
                (infeatures // features_per_int, outfeatures), dtype=torch.int32
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.empty(
                (math.ceil(infeatures / groupsize), outfeatures // features_per_int),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (math.ceil(infeatures / groupsize), outfeatures), dtype=torch.float16
            ),
        )
        if bias:
            self.register_buffer("bias", torch.empty(outfeatures, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        y = triton_matmul4(
            self.groupsize, x, self.qweight, self.scales, self.qzeros, self.bias
        )
        return y


@triton.jit
def matmul4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales_g,
    stride_scales_n,
    stride_zeros_g,
    stride_zeros_n,
    groupsize,
    NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = offs_am[:, None] < M

    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n

    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    if NO_GROUPS:

        scales = tl.load(scales_ptrs)
        zeros = tl.load(zeros_ptrs)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)

        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr = scales_ptrs + g_id * stride_scales_g

            scales = tl.load(ptr)
            ptr = zeros_ptrs + g_id * stride_zeros_g
            zeros = tl.load(ptr)

            zeros = (zeros >> zeros_shifter) & 0xF
            zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF

        b = b * scales[None, :] - zeros[None, :]

        b = b.to(tl.float16)

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul4(
    groupsize: int,
    a: torch.FloatTensor,
    qweight: torch.IntTensor,
    scales: torch.FloatTensor,
    qzeros: torch.IntTensor,
    bias: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:

    assert a.shape[-1] == (
        qweight.shape[0] * 8
    ), "A must be a multiple of 8 in the last dimension"
    assert a.is_contiguous(), "A must be contiguous"

    x = a.view(-1, a.shape[-1])

    M, K = a.shape
    N = qweight.shape[1]

    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 32
    GROUP_SIZE_M = 8

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul4_kernel[grid](
        x,
        qweight,
        c,
        scales,
        qzeros,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        c.stride(0),
        c.stride(1),
        scales.stride(0),
        scales.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        groupsize,
        groupsize == K,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    c = c.view(a.shape[:-1] + (N,))

    if bias is not None:
        c = c + bias

    return c
