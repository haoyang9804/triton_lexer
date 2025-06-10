import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cuda.language_extra import tid, st
from triton_dist import pynvshmem

from typing import Optional, List
from dataclasses import dataclass, field

from triton_dist.kernels.nvidia.common_ops import (
    set_signal,
    barrier_all_intra_node_non_atomic,
)
from triton_dist.kernels.nvidia.allgather import (
    AllGatherMethod,
    cp_engine_producer_all_gather_intra_node,
    get_auto_all_gather_method,
    cp_engine_producer_all_gather_inter_node,
)


@triton.jit(do_not_specialize=["rank"])
def copy_kernel(
    rank,
    local_buf_ptr,
    global_buf_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_id = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = sm_id // num_pid_n
    pid_n = sm_id % num_pid_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    data_ptr = (
        local_buf_ptr
        + (pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_local_m
        + (pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_local_n
    )
    dst_ptr = (
        global_buf_ptr
        + (rank * M_per_rank + pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_global_m
        + (pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_global_n
    )
    mask_data = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (
        pid_n * BLOCK_SIZE_N + offs_n[None, :] < N
    )
    mask_dst = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (
        pid_n * BLOCK_SIZE_N + offs_n[None, :] < N
    )

    data = tl.load(data_ptr, mask=mask_data)
    tl.store(dst_ptr, data, mask=mask_dst)


@triton.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "flag_value"])
def copy_and_barrier_all_intra_node_kernel(
    local_rank,
    rank,
    num_ranks,
    local_buf_ptr,
    global_buf_ptr,
    symm_barrier_ptr,
    symm_sync_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    flag_value,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    barrier_all_intra_node_non_atomic(
        local_rank, rank, num_ranks, symm_sync_ptr, flag_value
    )
    copy_kernel(
        rank,
        local_buf_ptr,
        global_buf_ptr,
        M_per_rank,
        N,
        stride_local_m,
        stride_local_n,
        stride_global_m,
        stride_global_n,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    thread_idx = tid(0)
    if thread_idx < num_ranks:
        st(symm_barrier_ptr + thread_idx, 1 if thread_idx == rank else 0)
    barrier_all_intra_node_non_atomic(
        local_rank, rank, num_ranks, symm_sync_ptr, flag_value + 1
    )


def local_copy_and_barrier_all(
    local_rank,
    rank,
    num_ranks,
    local_data,
    global_data,
    comm_buf,
    barrier_ptr,
    M_per_rank,
    N,
    phase,
    is_internode: bool = False,
):
    if not is_internode:
        grid = lambda META: (
            triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        copy_and_barrier_all_intra_node_kernel[grid](
            local_rank,
            rank,
            num_ranks,
            local_data,
            global_data,
            barrier_ptr,
            comm_buf,
            M_per_rank,
            N,
            local_data.stride(0),
            local_data.stride(1),
            global_data.stride(0),
            global_data.stride(1),
            phase,
            128,
            256,
        )

    else:
        pynvshmem.nvshmemx_barrier_all_on_stream(
            torch.cuda.current_stream().cuda_stream
        )
        barrier_ptr.fill_(0)
        grid = lambda META: (
            triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        copy_kernel[grid](
            rank,
            local_data,
            global_data,
            M_per_rank,
            N,
            local_data.stride(0),
            local_data.stride(1),
            global_data.stride(0),
            global_data.stride(1),
            128,
            256,
        )
        set_signal(
            barrier_ptr[rank].data_ptr(), 1, torch.cuda.current_stream(), is_internode
        )
        pynvshmem.nvshmemx_barrier_all_on_stream(
            torch.cuda.current_stream().cuda_stream
        )


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    ready_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    ready_value: tl.constexpr = 1,
    LOCAL_WORLD_SIZE: tl.constexpr = 8,
):

    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = rank // LOCAL_WORLD_SIZE
    nnodes = num_ranks // LOCAL_WORLD_SIZE

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

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            if nnodes == 1:
                alpha = 0
                beta = 0
                pid_m = (
                    pid_m + ((((rank ^ alpha) + beta) % num_ranks) * pid_ms_per_rank)
                ) % num_pid_m
            else:
                m_rank = pid_m // pid_ms_per_rank
                pid_m_intra_rank = pid_m - m_rank * pid_ms_per_rank
                m_node_id = m_rank // LOCAL_WORLD_SIZE
                m_local_rank = m_rank % LOCAL_WORLD_SIZE
                swizzle_m_node_id = (m_node_id + node_id) % nnodes
                swizzle_m_local_rank = (m_local_rank + rank) % LOCAL_WORLD_SIZE
                swizzle_m_rank = (
                    swizzle_m_node_id * LOCAL_WORLD_SIZE + swizzle_m_local_rank
                )

                pid_m = swizzle_m_rank * pid_ms_per_rank + pid_m_intra_rank

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            rank_beg = offs_am // M_per_rank
            rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank
            token = dl.wait(
                ready_ptr + rank_beg,
                rank_end - rank_beg + 1,
                "gpu",
                "acquire",
                waitValue=ready_value,
            )
            a_desc = dl.consume_token(a_desc, token)

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

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def _kernel_consumer_gemm_non_persistent_repr(proxy):
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

    return f"triton3x_sm{cap_major}{cap_minor}_ag_gemm_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}"


@triton.jit(
    do_not_specialize=["rank"],
    launch_metadata=_matmul_launch_metadata,
    repr=_kernel_consumer_gemm_non_persistent_repr,
)
def kernel_consumer_gemm_non_persistent(
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
    rank,
    WORLD_SIZE: tl.constexpr,
    barrier_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    c_dtype = c_ptr.dtype.element_ty

    tl.static_assert(a_dtype == b_dtype, "A and B must have the same dtype")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_per_rank = M // WORLD_SIZE
    m_offset = m_per_rank * rank
    pid_m_offset = tl.cdiv(m_offset, BLOCK_SIZE_M)
    pid_m = (pid_m + pid_m_offset) % num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M
    rank_beg = offs_am // m_per_rank
    rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // m_per_rank
    token = dl.wait(
        barrier_ptr + rank_beg, rank_end - rank_beg + 1, "gpu", "acquire", waitValue=1
    )

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    a_ptrs = dl.consume_token(a_ptrs, token)

    if a_dtype == tl.int8:
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
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, accumulator.to(c_dtype), mask=c_mask)


def matmul_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [3, 4]
        for w in [4, 8]
    ]


kernel_consumer_gemm_persistent_autotune = triton.autotune(
    configs=matmul_get_configs(), key=["M", "N", "K"]
)(kernel_consumer_gemm_persistent)


kernel_consumer_gemm_non_persistent_autotune = triton.autotune(
    configs=matmul_get_configs(), key=["M", "N", "K"]
)(kernel_consumer_gemm_non_persistent)


@dataclass
class AllGatherGEMMTensorParallelContext:

    max_M: int
    N_per_rank: int
    K: int
    tensor_dtype: torch.dtype

    rank: int
    num_ranks: int
    num_local_ranks: int
    is_multinode: bool = field(init=False)
    n_nodes: int = field(init=False)
    node_rank: int = field(init=False)
    local_rank: int = field(init=False)
    workspace_tensors: List[torch.Tensor] = field(init=False)
    barrier_tensors: List[torch.Tensor] = field(init=False)
    local_barrier_buff: List[torch.Tensor] = field(init=False)
    workspace_tensor: torch.Tensor = field(init=False)
    barrier_tensor: torch.Tensor = field(init=False)
    fake_barrier: torch.Tensor = field(init=False)
    comm_buf: torch.Tensor = field(init=False)
    intranode_barrier_dtype = torch.int32
    internode_barrier_dtype = torch.uint64
    barrier_target = 1

    gemm_stream: Optional[torch.cuda.streams.Stream] = None
    ag_intranode_stream: Optional[torch.cuda.streams.Stream] = None
    ag_internode_stream: Optional[torch.cuda.streams.Stream] = None

    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_SIZE_M: int = 8
    stages: int = 3
    warps: int = 8
    max_blocks: int = 1
    max_gemm_sm: int = field(init=False)
    phase: int = 1
    all_gather_method: AllGatherMethod = AllGatherMethod.Auto

    for_correctness: bool = False

    def __post_init__(self):
        self.is_multinode = self.num_ranks > self.num_local_ranks
        self.n_nodes = self.num_ranks // self.num_local_ranks
        self.node_rank = self.rank // self.num_local_ranks
        self.local_rank = self.rank % self.num_local_ranks
        self.workspace_tensors = pynvshmem.nvshmem_create_tensor_list_intra_node(
            [self.max_M, self.K], self.tensor_dtype
        )
        self.local_barrier_buff = pynvshmem.nvshmem_create_tensor(
            [self.max_blocks * self.num_ranks], self.intranode_barrier_dtype
        )
        self.comm_buf = pynvshmem.nvshmem_create_tensor(
            [3 * self.num_ranks], self.intranode_barrier_dtype
        )
        if not self.is_multinode:
            self.barrier_tensors = pynvshmem.nvshmem_create_tensor_list_intra_node(
                [self.num_ranks], self.intranode_barrier_dtype
            )
        else:
            self.barrier_tensors = pynvshmem.nvshmem_create_tensor_list_intra_node(
                [self.num_ranks], self.internode_barrier_dtype
            )
        self.workspace_tensor = self.workspace_tensors[self.local_rank]
        self.barrier_tensor = self.barrier_tensors[self.local_rank]
        self.comm_buf.fill_(0)
        self.barrier_tensor.fill_(0)

        self.fake_barrier = torch.ones(
            [self.num_ranks], dtype=self.intranode_barrier_dtype, device="cuda"
        )
        self.max_gemm_sm = torch.cuda.get_device_properties(
            "cuda"
        ).multi_processor_count

    def update(
        self,
        rank,
        num_ranks,
        num_local_ranks=8,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        stages=3,
        for_correctness=False,
        ag_stream=None,
        internode_ag_stream=None,
        gemm_stream=None,
    ):
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_local_ranks = num_local_ranks
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.stages = stages
        self.for_correctness = for_correctness
        self.ag_stream = ag_stream
        self.internode_ag_stream = internode_ag_stream
        self.gemm_stream = gemm_stream


def create_ag_gemm_context(
    tensor_A,
    tensor_B,
    rank,
    num_ranks,
    max_M,
    num_local_ranks=8,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    stages=3,
    ag_intranode_stream=None,
    ag_internode_stream=None,
    gemm_stream=None,
    for_correctness=False,
):

    M_per_rank, K = tensor_A.shape
    N_per_rank, _ = tensor_B.shape
    assert (
        tensor_A.shape[1] == tensor_B.shape[1]
    ), f"tensor_B should has shape (col_major) [{N_per_rank}, {K}], but get [{tensor_B.shape}]"
    assert (
        tensor_A.dtype == tensor_B.dtype
    ), f"Dtype of input and weight must be same: tensor_A dtype {tensor_A.dtype}, tensor_B dtype {tensor_B.dtype}"

    dtype = tensor_A.dtype
    gemm_stream = torch.cuda.Stream() if gemm_stream is None else gemm_stream
    ag_intranode_stream = (
        torch.cuda.Stream() if ag_intranode_stream is None else ag_intranode_stream
    )
    ag_internode_stream = (
        torch.cuda.Stream() if ag_internode_stream is None else ag_internode_stream
    )

    ctx = AllGatherGEMMTensorParallelContext(
        N_per_rank=N_per_rank,
        K=K,
        tensor_dtype=dtype,
        rank=rank,
        num_ranks=num_ranks,
        num_local_ranks=num_local_ranks,
        max_M=max_M,
        gemm_stream=gemm_stream,
        ag_intranode_stream=ag_intranode_stream,
        ag_internode_stream=ag_internode_stream,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        stages=stages,
        all_gather_method=get_auto_all_gather_method(num_ranks, num_local_ranks),
        for_correctness=for_correctness,
    )

    pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return ctx


def ag_gemm(
    a,
    b,
    ctx: AllGatherGEMMTensorParallelContext = None,
    rank=None,
    num_ranks=None,
    persistent=True,
    autotune=False,
):

    assert (
        a.shape[1] == b.shape[1]
    ), f"tensor_B should has shape (col_major) [{b.shape[0]}, {a.shape[1]}], but get [{b.shape}]"
    assert (
        a.dtype == b.dtype
    ), f"Dtype of input and weight must be same: tensor_A dtype {a.dtype}, tensor_B dtype {b.dtype}"

    M_per_rank, K = a.shape
    N_per_rank, _ = b.shape

    if ctx is None:
        assert rank is not None and num_ranks is not None
        M = M_per_rank * ctx.num_ranks
        ctx = create_ag_gemm_context(a, b, rank, num_ranks, max_M=M)

    assert (
        a.shape[0] * ctx.num_ranks <= ctx.max_M and a.shape[1] == ctx.K
    ), f"Shape of tensor_A must not exceed the maxmize M of ctx: tensor_A shape [{a.shape}], ctx shape [{ctx.max_M},{ctx.K}]"
    assert (
        b.shape[0] == ctx.N_per_rank
    ), f"N_per_rank of tensor_B must match that of ctx: tensor_B shape [{b.shape[0]}], ctx shape [{ctx.N_per_rank}]"
    assert (
        ctx.tensor_dtype == a.dtype
    ), f"dtype of ctx must match that of ctx: tensor_A dtype {a.dtype}, ctx dtype {ctx.tensor_dtype}"

    C = torch.empty(
        [ctx.num_ranks * M_per_rank, N_per_rank], dtype=a.dtype, device=a.device
    )

    local_copy_and_barrier_all(
        ctx.local_rank,
        ctx.rank,
        ctx.num_ranks,
        a,
        ctx.workspace_tensor,
        ctx.comm_buf,
        ctx.barrier_tensor,
        M_per_rank,
        K,
        ctx.phase,
        is_internode=ctx.is_multinode,
    )
    ctx.phase += 2

    rowise_ag_gemm_dispatcher(a, b, C, ctx, persistent=persistent, autotune=autotune)

    return C


def rowise_ag_gemm_dispatcher(
    a, b, c, ctx: AllGatherGEMMTensorParallelContext, persistent=False, autotune=False
):
    current_stream = torch.cuda.current_stream()
    if ctx.is_multinode:
        ctx.ag_internode_stream.wait_stream(current_stream)
    ctx.ag_intranode_stream.wait_stream(current_stream)
    ctx.gemm_stream.wait_stream(current_stream)

    if not ctx.is_multinode:
        cp_engine_producer_all_gather_intra_node(
            ctx.rank,
            ctx.num_ranks,
            a,
            ctx.workspace_tensors,
            ctx.barrier_tensors,
            ctx.ag_intranode_stream,
            for_correctness=ctx.for_correctness,
            all_gather_method=ctx.all_gather_method,
        )
    else:
        cp_engine_producer_all_gather_inter_node(
            a,
            ctx.workspace_tensors,
            ctx.barrier_tensors,
            ctx.barrier_target,
            ctx.rank,
            ctx.num_local_ranks,
            ctx.num_ranks,
            ctx.ag_intranode_stream,
            ctx.ag_internode_stream,
            for_correctness=ctx.for_correctness,
            all_gather_method=ctx.all_gather_method,
        )

    with torch.cuda.stream(ctx.gemm_stream):
        M_per_rank, K = a.shape
        M = M_per_rank * ctx.num_ranks
        if not persistent:
            grid = lambda META: (
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(ctx.N_per_rank, META["BLOCK_SIZE_N"]),
            )
            if not autotune:
                compiled = kernel_consumer_gemm_non_persistent[grid](
                    ctx.workspace_tensor[:M],
                    b,
                    c,
                    M,
                    ctx.N_per_rank,
                    ctx.K,
                    ctx.workspace_tensor.stride(0),
                    ctx.workspace_tensor.stride(1),
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    ctx.rank,
                    ctx.num_ranks,
                    ctx.barrier_tensor,
                    ctx.BLOCK_M,
                    ctx.BLOCK_N,
                    ctx.BLOCK_K,
                    ctx.GROUP_SIZE_M,
                    num_stages=ctx.stages,
                    num_warps=ctx.warps,
                )
            else:
                compiled = kernel_consumer_gemm_non_persistent_autotune[grid](
                    ctx.workspace_tensor[:M],
                    b,
                    c,
                    M,
                    ctx.N_per_rank,
                    ctx.K,
                    ctx.workspace_tensor.stride(0),
                    ctx.workspace_tensor.stride(1),
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    ctx.rank,
                    ctx.num_ranks,
                    ctx.barrier_tensor,
                )
        else:

            def alloc_fn(size: int, alignment: int, stream: Optional[int]):
                return torch.empty(size, device="cuda", dtype=torch.int8)

            triton.set_allocator(alloc_fn)

            internode_ag_sm = ctx.n_nodes - 1
            gemm_sm = ctx.max_gemm_sm - internode_ag_sm
            grid = lambda META: (
                min(
                    gemm_sm,
                    triton.cdiv(M, META["BLOCK_SIZE_M"])
                    * triton.cdiv(ctx.N_per_rank, META["BLOCK_SIZE_N"]),
                ),
            )

            if not autotune:
                compiled = kernel_consumer_gemm_persistent[grid](
                    ctx.workspace_tensor[:M],
                    b,
                    c,
                    M,
                    ctx.N_per_rank,
                    ctx.K,
                    ctx.rank,
                    ctx.num_ranks,
                    ctx.barrier_tensor,
                    ctx.BLOCK_M,
                    ctx.BLOCK_N,
                    ctx.BLOCK_K,
                    ctx.GROUP_SIZE_M,
                    False,
                    gemm_sm,
                    ready_value=ctx.barrier_target,
                    LOCAL_WORLD_SIZE=ctx.num_local_ranks,
                    num_stages=ctx.stages,
                    num_warps=ctx.warps,
                )
            else:
                compiled = kernel_consumer_gemm_persistent_autotune[grid](
                    ctx.workspace_tensor[:M],
                    b,
                    c,
                    M,
                    ctx.N_per_rank,
                    ctx.K,
                    ctx.rank,
                    ctx.num_ranks,
                    ctx.barrier_tensor,
                    LOCAL_WORLD_SIZE=ctx.num_local_ranks,
                    EPILOGUE_SUBTILE=False,
                    NUM_SMS=gemm_sm,
                )

    if ctx.is_multinode:
        current_stream.wait_stream(ctx.ag_internode_stream)
    current_stream.wait_stream(ctx.ag_intranode_stream)
    current_stream.wait_stream(ctx.gemm_stream)

    return compiled


def gemm_persistent(a, b, ctx: AllGatherGEMMTensorParallelContext, autotune=False):
    M, K = a.shape
    N, _ = b.shape
    C = torch.empty([M, N], dtype=a.dtype, device=a.device)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    if not autotune:
        kernel_consumer_gemm_persistent[grid](
            a,
            b,
            C,
            M,
            N,
            K,
            ctx.rank,
            ctx.num_ranks,
            ctx.fake_barrier_tensor,
            ctx.comm_buf,
            ctx.BLOCK_M,
            ctx.BLOCK_N,
            ctx.BLOCK_K,
            8,
            False,
            NUM_SMS=NUM_SMS,
            num_stages=ctx.stages,
            num_warps=8,
        )
    else:
        kernel_consumer_gemm_persistent_autotune[grid](
            a,
            b,
            C,
            M,
            N,
            K,
            ctx.rank,
            ctx.num_ranks,
            ctx.fake_barrier_tensor,
            ctx.comm_buf,
            EPILOGUE_SUBTILE=False,
            NUM_SMS=NUM_SMS,
        )

    return C


def gemm_non_persistent(a, b, ctx: AllGatherGEMMTensorParallelContext):
    M, K = a.shape
    N, _ = b.shape
    C = torch.empty([M, N], dtype=a.dtype, device=a.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    if not ctx.autotune:
        kernel_consumer_gemm_non_persistent[grid](
            a,
            b,
            C,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(0),
            C.stride(0),
            C.stride(1),
            ctx.rank,
            ctx.num_ranks,
            ctx.fake_barrier_tensor,
            ctx.BLOCK_M,
            ctx.BLOCK_N,
            ctx.BLOCK_K,
            8,
            num_stages=ctx.stages,
            num_warps=8,
        )
    else:
        kernel_consumer_gemm_persistent_autotune[grid](
            a,
            b,
            C,
            M,
            N,
            K,
            ctx.rank,
            ctx.num_ranks,
            ctx.fake_barrier_tensor,
            ctx.comm_buf,
            EPILOGUE_SUBTILE=False,
            NUM_SMS=0,
        )

    return C
