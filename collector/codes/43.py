import dataclasses
from typing import List, Optional

import torch

import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cuda.language_extra import __syncthreads, atomic_add, tid
from triton_dist import pynvshmem
from triton_dist.kernels.nvidia.reduce_scatter import (
    ReduceScatter2DContext,
    create_reduce_scater_2d_ctx,
    reduce_scatter_2d_op,
)
from triton_dist.kernels.nvidia.gemm_rs_threadblock_swizzle import (
    threadblock_swizzle_gemm_reduce_scatter_kernel,
)


@dataclasses.dataclass
class GEMMReduceScatterTensorParallelContext:
    rs_ctx: ReduceScatter2DContext
    output_dtype: torch.dtype

    gemm_out_bufs: List[torch.Tensor]

    rs_stream: torch.cuda.Stream

    num_gemm_sms: int
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_M: int = 8
    stages: int = 3

    def update(
        self,
        rs_stream,
        output_dtype=None,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_M=8,
        stages=3,
    ):
        self.rs_stream = rs_stream
        self.output_dtype = output_dtype
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_M = GROUP_M
        self.stages = stages

    def get_gemm_out_buf(self, input):
        M, _ = input.shape
        local_rank = self.rs_ctx.local_rank
        return self.gemm_out_bufs[local_rank][:M]


def create_gemm_rs_context(
    max_M,
    N,
    rank,
    world_size,
    local_world_size,
    output_dtype,
    rs_stream,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    GROUP_M=8,
    stages=3,
) -> GEMMReduceScatterTensorParallelContext:
    rs_ctx = create_reduce_scater_2d_ctx(
        max_M,
        N,
        rank,
        world_size,
        local_world_size,
        output_dtype,
        overlap_with_gemm=True,
    )
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = NUM_SMS - rs_ctx.num_rs_sms
    gemm_out_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node(
        [max_M, N], output_dtype
    )
    ctx = GEMMReduceScatterTensorParallelContext(
        rs_ctx=rs_ctx,
        output_dtype=output_dtype,
        gemm_out_bufs=gemm_out_bufs,
        rs_stream=rs_stream,
        num_gemm_sms=num_gemm_sms,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        stages=stages,
    )
    return ctx


@triton.jit
def swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = args["a_ptr"].element_size()
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def _gemm_rs_persistent_repr(proxy):
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = (
        constexprs["BLOCK_SIZE_M"],
        constexprs["BLOCK_SIZE_N"],
        constexprs["BLOCK_SIZE_K"],
    )

    return f"triton3x_sm{cap_major}{cap_minor}_gemm_rs_persistent_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_ntn"


@triton.jit(launch_metadata=_matmul_launch_metadata, repr=_gemm_rs_persistent_repr)
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    barrier_ptr,
    counter_ptr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):

    rank = dl.rank()
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // WORLD_SIZE

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pid_m_offset = (rank + 1) * M_per_rank // BLOCK_SIZE_M

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            pid_m, pid_n = swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)
            if NNODES != 1:
                pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(
                    pid_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M
                )
            else:
                pid_m = (pid_m + pid_m_offset) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

            counter_start = offs_am // M_per_rank
            counter_end = (offs_am + BLOCK_SIZE_M - 1) // M_per_rank
            counter_end = min(counter_end, WORLD_SIZE - 1)
            for counter_id in range(counter_start, counter_end + 1):
                m_start = M_per_rank * counter_id
                m_end = M_per_rank * (counter_id + 1) - 1
                tiled_m_start = m_start // BLOCK_SIZE_M
                tiled_m_end = m_end // BLOCK_SIZE_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                val = tl.atomic_add(
                    counter_ptr + counter_id, 1, sem="release", scope="gpu"
                )
                if val == tiled_m_size * num_pid_n - 1:
                    dl.notify(
                        barrier_ptr + counter_id, rank, signal=1, comm_scope="gpu"
                    )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def _gemm_rs_non_persistent_repr(proxy):
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = (
        constexprs["BLOCK_SIZE_M"],
        constexprs["BLOCK_SIZE_N"],
        constexprs["BLOCK_SIZE_K"],
    )
    if constexprs.get("stride_am", None) == 1:
        a_trans = "n"
    elif constexprs.get("stride_ak", None) == 1:
        a_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    if constexprs.get("stride_bk", None) == 1:
        b_trans = "n"
    elif constexprs.get("stride_bn", None) == 1:
        b_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    if constexprs.get("stride_cm", None) == 1:
        c_trans = "n"
    elif constexprs.get("stride_cn", None) == 1:
        c_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    return f"triton3x_sm{cap_major}{cap_minor}_gemm_rs_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}"


@triton.jit(launch_metadata=_matmul_launch_metadata, repr=_gemm_rs_non_persistent_repr)
def kernel_gemm_rs_producer_non_persistent(
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
    barrier_ptr,
    counter_ptr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    tl.static_assert(a_ptr.dtype.is_ptr(), "A should be a pointer")
    tl.static_assert(b_ptr.dtype.is_ptr(), "B should be a pointer")
    tl.static_assert(c_ptr.dtype.is_ptr(), "C should be a pointer")
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    tl.static_assert(a_dtype == b_dtype, "A and B should have the same dtype")

    rank = dl.rank()
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    M_per_rank = M // WORLD_SIZE

    pid_m, pid_n = swizzle_2d(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    if NNODES != 1:
        pid_m = threadblock_swizzle_gemm_reduce_scatter_kernel(
            pid_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M
        )
    else:
        pid_m_offset = (rank + 1) * M_per_rank // BLOCK_SIZE_M
        pid_m = (pid_m + pid_m_offset) % num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if a_ptr.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, accumulator, mask=out_mask)

    segment_start = pid_m * BLOCK_SIZE_M // M_per_rank
    segment_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // M_per_rank
    __syncthreads()
    segment = segment_start + tid(axis=0)
    if segment <= segment_end:
        m_start = M_per_rank * segment
        m_end = M_per_rank * (segment + 1) - 1
        tiled_m_start = m_start // BLOCK_SIZE_M
        tiled_m_end = m_end // BLOCK_SIZE_M
        tiled_m_size = tiled_m_end - tiled_m_start + 1
        val = atomic_add(counter_ptr + segment, 1, semantic="release", scope="gpu")
        if val == num_pid_n * tiled_m_size - 1:
            atomic_add(barrier_ptr + segment, 1, semantic="release", scope="gpu")


def gemm_rs_producer_persistent(
    a,
    b,
    c,
    barrier,
    workspace,
    world_size,
    local_world_size,
    num_gemm_sms,
    gemm_stream,
    triton_config: triton.Config,
):

    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    current_stream = torch.cuda.current_stream()
    gemm_stream.wait_stream(current_stream)

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (
        min(
            num_gemm_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    with torch.cuda.stream(gemm_stream):
        compiled = kernel_gemm_rs_producer_persistent[grid](
            a,
            b,
            c,
            M,
            N,
            local_K,
            barrier,
            workspace,
            local_world_size,
            world_size,
            **triton_config.all_kwargs(),
        )

    current_stream.wait_stream(gemm_stream)

    return compiled


def gemm_rs_producer_non_persistent(
    a,
    b,
    c,
    barrier,
    workspace,
    world_size,
    local_world_size,
    gemm_stream,
    triton_config: triton.Config,
):

    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, local_K = a.shape
    N, local_K = b.shape

    current_stream = torch.cuda.current_stream()
    gemm_stream.wait_stream(current_stream)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    with torch.cuda.stream(gemm_stream):
        compiled = kernel_gemm_rs_producer_non_persistent[grid](
            a,
            b,
            c,
            M,
            N,
            local_K,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(0),
            c.stride(0),
            c.stride(1),
            barrier,
            workspace,
            local_world_size,
            world_size,
            **triton_config.all_kwargs(),
        )

    current_stream.wait_stream(gemm_stream)

    return compiled


def padded_to_BLOCK_M(input, world_size, BLOCK_SIZE_M):
    M, local_K = input.shape

    M_per_rank = M // world_size
    pad_size = (M_per_rank + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * BLOCK_SIZE_M
    if pad_size == M_per_rank:
        return input
    input = input.reshape(world_size, M_per_rank, local_K)
    pad_input = torch.empty(
        (world_size, pad_size, local_K), dtype=input.dtype, device=input.device
    )
    pad_input[:, :M_per_rank].copy_(input)
    pad_input = pad_input.reshape(-1, local_K)
    return pad_input


def update_triton_config(
    M, N, K, dtype: torch.dtype, world_size, local_world_size, config: triton.Config
):

    from triton_dist.kernels.nvidia.comm_perf_model import (
        estimate_reduce_scatter_time,
        get_nic_bandwidth_per_gpu,
    )
    from triton_dist.kernels.nvidia.gemm_perf_model import estimate_gemm_sol_time_ms
    from triton_dist.utils import get_intranode_max_speed

    gemm_time_ms = estimate_gemm_sol_time_ms(M, N, K, dtype)
    rs_time_ms = (
        estimate_reduce_scatter_time(
            M * N * dtype.itemsize,
            world_size,
            local_world_size,
            get_intranode_max_speed(),
            get_nic_bandwidth_per_gpu(),
        )
        * 1000
    )
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    GROUP_SIZE_M = config.kwargs["GROUP_SIZE_M"]
    if gemm_time_ms < rs_time_ms:

        M_per_rank = M // world_size
        tiled_m_per_rank = (M_per_rank + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        if tiled_m_per_rank < GROUP_SIZE_M:

            config.kwargs["GROUP_SIZE_M"] = tiled_m_per_rank

    return config


def gemm_rs_op(
    input, weight, ctx: GEMMReduceScatterTensorParallelContext, persistent: bool = True
):
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    output_dtype = ctx.output_dtype
    num_gemm_sms = ctx.num_gemm_sms

    orig_M = input.shape[0]
    orig_M_per_rank = orig_M // world_size
    M, local_K = input.shape
    N = weight.shape[0]
    assert N == ctx.rs_ctx.N

    assert M % world_size == 0
    assert weight.shape[1] == local_K
    local_M = M // world_size
    current_stream = torch.cuda.current_stream()
    rs_stream.wait_stream(current_stream)

    output = torch.empty((local_M, N), dtype=output_dtype, device=input.device)
    workspace = torch.zeros((world_size,), dtype=torch.int32, device=input.device)
    gemm_out = ctx.get_gemm_out_buf(input)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf

    if persistent:
        triton_config = triton.Config(
            {
                "BLOCK_SIZE_M": ctx.BLOCK_M,
                "BLOCK_SIZE_N": ctx.BLOCK_N,
                "BLOCK_SIZE_K": ctx.BLOCK_K,
                "GROUP_SIZE_M": ctx.GROUP_M,
                "NUM_SMS": num_gemm_sms,
                "EPILOGUE_SUBTILE": False,
            },
            num_stages=ctx.stages,
            num_warps=8,
        )
        gemm_rs_producer_persistent(
            input,
            weight,
            gemm_out,
            scatter_signal,
            workspace,
            world_size,
            local_world_size,
            num_gemm_sms,
            current_stream,
            triton_config,
        )
    else:
        triton_config = triton.Config(
            {
                "BLOCK_SIZE_M": ctx.BLOCK_M,
                "BLOCK_SIZE_N": ctx.BLOCK_N,
                "BLOCK_SIZE_K": ctx.BLOCK_K,
                "GROUP_SIZE_M": ctx.GROUP_M,
            },
            num_stages=ctx.stages,
            num_warps=8,
        )
        triton_config = update_triton_config(
            M, N, local_K, input.dtype, world_size, local_world_size, triton_config
        )
        gemm_rs_producer_non_persistent(
            input,
            weight,
            gemm_out,
            scatter_signal,
            workspace,
            world_size,
            local_world_size,
            current_stream,
            triton_config,
        )

    with torch.cuda.stream(rs_stream):
        output = reduce_scatter_2d_op(gemm_out, ctx.rs_ctx)
    current_stream.wait_stream(rs_stream)

    return output[:orig_M_per_rank]


def gemm_rs(a, b, ctx, persistent=True):

    c = gemm_rs_op(a, b, ctx, persistent)
    return c
