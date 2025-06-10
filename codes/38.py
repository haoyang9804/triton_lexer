from dataclasses import dataclass, field
import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from typing import Optional, List
from triton._C.libtriton_distributed.distributed import moe_ag_scatter_align_block_size
from triton_dist.kernels.nvidia.common_ops import set_signal
from triton_dist.kernels.nvidia.allgather import (
    AllGatherMethod,
    cp_engine_producer_all_gather_intra_node,
    get_auto_all_gather_method,
    cp_engine_producer_all_gather_inter_node,
)

from triton_dist import pynvshmem


@dataclass
class MoEInfo:
    num_experts: int = None
    topk: int = None
    topked_num_tokens: int = None
    sorted_topk_ids: torch.Tensor = None
    aligned_expert_ids: torch.Tensor = None
    aligned_barrier_ids: torch.Tensor = None
    padded_num_tokens: torch.Tensor = None


@dataclass
class MoEAllGatherGroupGEMMTensorParallelContext:

    max_M: int
    N_per_rank: int
    K: int
    tensor_dtype: torch.dtype

    rank: int
    num_ranks: int
    num_local_ranks: int = 8
    is_multinode: bool = field(init=False)
    n_nodes: int = field(init=False)
    node_rank: int = field(init=False)
    local_rank: int = field(init=False)

    workspace_tensors: List[torch.Tensor] = field(init=False)
    barrier_tensors: List[torch.Tensor] = field(init=False)
    workspace_tensor: torch.Tensor = field(init=False)
    barrier_tensor: torch.Tensor = field(init=False)
    intranode_barrier_dtype = torch.int32
    internode_barrier_dtype = torch.uint64
    barrier_target = 1

    group_gemm_stream: Optional[torch.cuda.streams.Stream] = None
    ag_intranode_stream: Optional[torch.cuda.streams.Stream] = None
    ag_internode_stream: Optional[torch.cuda.streams.Stream] = None

    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_SIZE_M: int = 8
    stages: int = 3
    warps: int = 8
    moe_info: MoEInfo = None
    all_gather_method: AllGatherMethod = AllGatherMethod.Auto

    def __post_init__(self):
        self.is_multinode = self.num_ranks > self.num_local_ranks
        self.n_nodes = self.num_ranks // self.num_local_ranks
        self.node_rank = self.rank // self.num_local_ranks
        self.local_rank = self.rank % self.num_local_ranks
        self.workspace_tensors = pynvshmem.nvshmem_create_tensor_list_intra_node(
            [self.max_M, self.K], self.tensor_dtype
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

    @staticmethod
    def sort_topk_ids_align_block_size(
        topk_ids: torch.Tensor,
        num_experts: int,
        num_ranks: int,
        num_tokens_per_rank: int,
        block_size: int,
    ):

        num_topk = topk_ids.shape[1]
        sorted_ids = torch.empty(
            (
                (num_tokens_per_rank * num_topk + num_experts * (block_size - 1))
                * num_ranks,
            ),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        expert_ids = torch.empty(
            ((num_tokens_per_rank * num_topk + num_experts) * num_ranks,),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        block_barrier_ids = torch.empty(
            ((num_tokens_per_rank * num_topk + num_experts) * num_ranks,),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        rank_block_num = torch.empty(
            num_ranks,
            dtype=torch.int32,
            device=topk_ids.device,
        )
        sorted_ids.fill_(topk_ids.numel())
        num_tokens_post_pad = torch.empty(
            (1), dtype=torch.int32, device=topk_ids.device
        )

        moe_ag_scatter_align_block_size(
            topk_ids,
            num_experts,
            num_ranks,
            num_tokens_per_rank * topk_ids.shape[1],
            block_size,
            sorted_ids,
            expert_ids,
            block_barrier_ids,
            rank_block_num,
            num_tokens_post_pad,
            torch.cuda.current_stream().cuda_stream,
        )

        return (
            sorted_ids,
            expert_ids,
            block_barrier_ids,
            rank_block_num,
            num_tokens_post_pad,
        )

    def local_copy_and_reset_barrier(self, local_data):

        M_per_rank = local_data.shape[0]
        pynvshmem.nvshmemx_barrier_all_on_stream(
            torch.cuda.current_stream().cuda_stream
        )
        self.barrier_tensor.zero_()
        dst = self.workspace_tensor[
            self.rank * M_per_rank : (self.rank + 1) * M_per_rank, :
        ]
        dst.copy_(local_data)
        set_signal(
            self.barrier_tensor[self.rank].data_ptr(),
            1,
            torch.cuda.current_stream(),
            self.is_multinode,
        )
        pynvshmem.nvshmemx_barrier_all_on_stream(
            torch.cuda.current_stream().cuda_stream
        )

    def update_topk_id(self, local_data, full_topk_ids, num_experts):
        num_tokens, topk = full_topk_ids.shape
        M_per_rank = local_data.shape[0]
        assert (
            num_tokens == M_per_rank * self.num_ranks
        ), f"num tokens must equal to M_per_rank * num_ranks of the ctx: num tokens={num_tokens}, M_per_rank * num_ranks={M_per_rank*self.num_ranks}"

        (
            full_sorted_token_ids,
            full_token_expert_ids,
            block_wait_barriers,
            _,
            full_num_tokens_post_padded_list,
        ) = self.sort_topk_ids_align_block_size(
            full_topk_ids, num_experts, self.num_ranks, M_per_rank, self.BLOCK_M
        )
        self._update_moe_info(
            num_experts,
            topk,
            full_topk_ids.numel(),
            full_sorted_token_ids,
            full_token_expert_ids,
            block_wait_barriers,
            full_num_tokens_post_padded_list,
        )

    def _update_moe_info(
        self,
        num_experts,
        topk,
        topked_num_tokens,
        sorted_topk_ids,
        expert_ids_per_block,
        barrier_ids_per_block,
        padded_num_tokens,
    ):

        self.moe_info.num_experts = num_experts
        self.moe_info.topk = topk
        self.moe_info.topked_num_tokens = topked_num_tokens
        self.moe_info.sorted_topk_ids = sorted_topk_ids
        self.moe_info.aligned_expert_ids = expert_ids_per_block
        self.moe_info.aligned_barrier_ids = barrier_ids_per_block
        self.moe_info.padded_num_tokens = padded_num_tokens


def create_ag_group_gemm_context(
    tensor_A,
    tensor_B,
    rank,
    num_ranks,
    full_topk_ids,
    max_M,
    ag_intranode_stream=None,
    ag_internode_stream=None,
    group_gemm_stream=None,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    GROUP_SIZE_M=8,
    stages=3,
    warps=8,
    num_local_ranks=8,
):

    M_per_rank, K = tensor_A.shape
    num_experts, K, N_per_rank = tensor_B.shape
    assert tensor_A.shape[1] == tensor_B.shape[1]

    group_gemm_stream = (
        torch.cuda.Stream() if group_gemm_stream is None else group_gemm_stream
    )
    ag_intranode_stream = (
        torch.cuda.Stream() if ag_intranode_stream is None else ag_intranode_stream
    )
    ag_internode_stream = (
        torch.cuda.Stream() if ag_internode_stream is None else ag_internode_stream
    )

    ctx = MoEAllGatherGroupGEMMTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        num_local_ranks=num_local_ranks,
        max_M=max_M,
        N_per_rank=N_per_rank,
        K=K,
        tensor_dtype=tensor_A.dtype,
        group_gemm_stream=group_gemm_stream,
        ag_intranode_stream=ag_intranode_stream,
        ag_internode_stream=ag_internode_stream,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        stages=stages,
        warps=warps,
        all_gather_method=get_auto_all_gather_method(num_ranks, num_local_ranks),
        moe_info=MoEInfo(),
    )
    ctx.update_topk_id(tensor_A, full_topk_ids, num_experts)

    pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return ctx


def ag_group_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    ctx: MoEAllGatherGroupGEMMTensorParallelContext = None,
    rank=None,
    num_ranks=None,
    full_topk_ids=None,
):

    ntokens, hidden = a.shape
    num_experts, h, N_per_rank = b.shape
    assert (
        hidden == h == ctx.K
    ), f"dim hidden does not match: A<ntokens, hidden> and B<nexperts, hidden, N_per_rank> : {a.shape} vs {b.shape} vs {ctx.K}"
    assert (
        a.dtype == b.dtype
    ), f"Dtype of input and weight must be same: tensor_A dtype {a.dtype}, tensor_B dtype {b.dtype}"
    assert a.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ], f"{a.dtype} not supported"

    if ctx is None:
        assert full_topk_ids is not None
        assert rank is not None and num_ranks is not None
        M = ntokens * num_ranks
        ctx = create_ag_group_gemm_context(
            a, b, rank, num_ranks, full_topk_ids, max_M=M
        )

    assert (
        ntokens * ctx.num_ranks <= ctx.max_M
    ), f"Shape of Allgathered tensor_A must not exceed max_M of ctx: tensor_A shape [{ntokens * ctx.num_ranks}], ctx max_M [{ctx.max_M}]"
    assert (
        N_per_rank == ctx.N_per_rank
    ), f"N_per_rank of tensor_B must match that of ctx: tensor_B shape [{b.shape[2]}], ctx shape [{ctx.N_per_rank}]"
    assert (
        ctx.tensor_dtype == a.dtype
    ), f"dtype of ctx must match that of ctx: tensor_A dtype {a.dtype}, ctx dtype {ctx.tensor_dtype}"

    c = torch.empty(
        [ctx.moe_info.topk * ntokens * ctx.num_ranks, ctx.N_per_rank],
        dtype=ctx.tensor_dtype,
        device=a.device,
    )

    rowise_ag_scatter_group_gemm_dispatcher(a, b, c, ctx)

    return c


def rowise_ag_scatter_group_gemm_dispatcher(
    a, b, c, ctx: MoEAllGatherGroupGEMMTensorParallelContext
):
    ctx.local_copy_and_reset_barrier(a)

    current_stream = torch.cuda.current_stream()
    if ctx.is_multinode:
        ctx.ag_internode_stream.wait_stream(current_stream)
    ctx.ag_intranode_stream.wait_stream(current_stream)
    ctx.group_gemm_stream.wait_stream(current_stream)

    if not ctx.is_multinode:
        cp_engine_producer_all_gather_intra_node(
            ctx.rank,
            ctx.num_ranks,
            a,
            ctx.workspace_tensors,
            ctx.barrier_tensors,
            ctx.ag_intranode_stream,
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
            all_gather_method=ctx.all_gather_method,
        )

    with torch.cuda.stream(ctx.group_gemm_stream):
        EM = ctx.moe_info.sorted_topk_ids.shape[0]
        M_per_rank, K = a.shape
        M = M_per_rank * ctx.num_ranks
        local_ag_buffer = ctx.workspace_tensor[:M]

        grid = lambda META: (
            triton.cdiv(EM, META["BLOCK_SIZE_M"])
            * triton.cdiv(ctx.N_per_rank, META["BLOCK_SIZE_N"]),
        )
        compiled = kernel_consumer_m_parallel_scatter_group_gemm[grid](
            local_ag_buffer,
            b,
            c,
            ctx.barrier_tensor,
            ctx.moe_info.sorted_topk_ids,
            ctx.moe_info.aligned_expert_ids,
            ctx.moe_info.padded_num_tokens,
            ctx.moe_info.aligned_barrier_ids,
            ctx.moe_info.topked_num_tokens,
            EM,
            ctx.N_per_rank,
            ctx.K,
            local_ag_buffer.stride(0),
            local_ag_buffer.stride(1),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
            ctx.BLOCK_M,
            ctx.BLOCK_N,
            ctx.BLOCK_K,
            ctx.GROUP_SIZE_M,
            ctx.moe_info.topk,
            ctx.rank,
            ctx.num_ranks,
            num_stages=ctx.stages,
            num_warps=ctx.warps,
        )

    if ctx.is_multinode:
        current_stream.wait_stream(ctx.ag_internode_stream)
    current_stream.wait_stream(ctx.ag_intranode_stream)
    current_stream.wait_stream(ctx.group_gemm_stream)

    return compiled


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

    return f"triton3x_sm{cap_major}{cap_minor}_ag_group_gemm_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}"


@triton.jit(do_not_specialize=["rank"], repr=_kernel_consumer_gemm_non_persistent_repr)
def kernel_consumer_m_parallel_scatter_group_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    block_barrier_ptr,
    sorted_token_ids_ptr,
    token_expert_ids_ptr,
    num_tokens_post_padded,
    block_barrier_id_ptr,
    num_valid_tokens,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    TOP_K: tl.constexpr,
    rank,
    WORLD_SIZE: tl.constexpr,
    SWIZZLE_OFFSET: tl.constexpr = 3,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_blocks_per_group = GROUP_SIZE_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    m_per_rank = num_block_m // WORLD_SIZE
    m_offset = m_per_rank * ((rank + SWIZZLE_OFFSET) % WORLD_SIZE)
    pid_m = (pid_m + m_offset) % num_block_m

    num_tokens_post_padded_value = tl.load(num_tokens_post_padded)

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded_value:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr + offs_token[:, None] // TOP_K * stride_am + offs_k[None, :] * stride_ak
    )

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(token_expert_ids_ptr + pid_m)

    b_ptrs = (
        b_ptr
        + offs_be * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    offs_barrier = tl.load(block_barrier_id_ptr + pid_m)
    token = dl.wait(block_barrier_ptr + offs_barrier, 1, "gpu", "acquire")
    a_ptrs = dl.consume_token(a_ptrs, token)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        )
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
