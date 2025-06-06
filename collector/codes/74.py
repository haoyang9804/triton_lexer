import torch
import dataclasses
import triton
import triton_dist
import triton.language as tl
import triton_dist.language as dl

from typing import Optional, List
from triton_dist import pynvshmem
from triton_dist.kernels.nvidia.common_ops import (
    BarrierAllContext,
    wait_eq,
    barrier_all_on_stream,
)
from triton_dist.kernels.nvidia.reduce_scatter import ring_reduce
from triton.language.extra import libshmem_device

import os

SIGNAL_DTYPE = torch.uint64


@dataclasses.dataclass
class ReduceScatter2DContext:
    max_M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype
    overlap_with_gemm: bool

    scatter_bufs: List[torch.Tensor]
    rs_per_node_bufs: List[torch.Tensor]
    p2p_bufs: List[torch.Tensor]

    signal_bufs: List[torch.Tensor]
    barrier: BarrierAllContext

    reduction_stream: torch.cuda.Stream
    p2p_stream: torch.cuda.Stream

    num_sync_sms: int
    num_p2p_sms: int
    num_reduction_sms: int

    scatter_signal_bufs: List[torch.Tensor] = dataclasses.field(init=False)
    rs_per_node_signal_bufs: List[torch.Tensor] = dataclasses.field(init=False)

    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)

    scatter_signal_buf_list_for_each_node: List[torch.Tensor] = dataclasses.field(
        init=False
    )

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        self.scatter_signal_buf_list_for_each_node = []
        for buf in self.signal_bufs:
            assert buf.shape[0] >= 2 * self.world_size

        self.scatter_signal_bufs = [buf[: self.world_size] for buf in self.signal_bufs]
        self.rs_per_node_signal_bufs = [
            buf[self.world_size : self.world_size * 2] for buf in self.signal_bufs
        ]

        for node_id in range(self.nnodes):
            self.scatter_signal_buf_list_for_each_node.append(
                self.scatter_signal_bufs[self.local_rank][
                    node_id
                    * self.local_world_size : (node_id + 1)
                    * self.local_world_size
                ]
            )

    def reset_barriers(self) -> int:
        self.signal_bufs[self.local_rank].fill_(0)

    def get_scatter_bufs_and_signal_for_each_node(self, input, node_id):
        M = input.shape[0]
        M_per_rank = M // self.world_size
        M_per_node = M_per_rank * self.local_world_size
        scatter_bufs_intra_node = [
            self.scatter_bufs[i][node_id * M_per_node : (node_id + 1) * M_per_node]
            for i in range(self.local_world_size)
        ]
        return (
            scatter_bufs_intra_node,
            self.scatter_signal_buf_list_for_each_node[node_id],
        )

    @property
    def rs_per_node_buf(self) -> torch.Tensor:
        return self.rs_per_node_bufs[self.local_rank]

    @property
    def rs_per_node_signal_buf(self) -> torch.Tensor:
        return self.rs_per_node_signal_bufs[self.local_rank]

    @property
    def p2p_buf(self) -> torch.Tensor:
        return self.p2p_bufs[self.local_rank]

    @property
    def num_rs_sms(self) -> int:
        if self.nnodes > 1:
            return self.num_sync_sms + self.num_p2p_sms + self.num_reduction_sms
        else:

            return 0

    @property
    def scatter_signal_buf(self) -> torch.Tensor:
        return self.scatter_signal_bufs[self.local_rank]


def create_reduce_scater_2d_ctx(
    max_M,
    N,
    rank,
    world_size,
    local_world_size,
    dtype,
    overlap_with_gemm=True,
    num_reduction_sms=15,
) -> ReduceScatter2DContext:

    assert world_size % local_world_size == 0
    assert max_M % world_size == 0

    scatter_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, N], dtype)

    rs_per_node_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node(
        [max_M // local_world_size, N], dtype
    )

    p2p_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node(
        [max_M // local_world_size, N], dtype
    )

    num_signal_bufs = 2
    signal_bufs = pynvshmem.nvshmem_create_tensor_list_intra_node(
        [
            world_size * num_signal_bufs,
        ],
        SIGNAL_DTYPE,
    )

    barrier_all_on_stream(None, torch.cuda.current_stream())

    p2p_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
    reduction_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

    num_sync_sms = 0
    num_p2p_sms = 1
    ctx = ReduceScatter2DContext(
        max_M=max_M,
        N=N,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        dtype=dtype,
        overlap_with_gemm=overlap_with_gemm,
        scatter_bufs=scatter_bufs,
        rs_per_node_bufs=rs_per_node_bufs,
        p2p_bufs=p2p_bufs,
        signal_bufs=signal_bufs,
        barrier=BarrierAllContext(True),
        reduction_stream=reduction_stream,
        p2p_stream=p2p_stream,
        num_sync_sms=num_sync_sms,
        num_p2p_sms=num_p2p_sms,
        num_reduction_sms=num_reduction_sms,
    )
    return ctx


@triton.jit
def kernel_inter_node_p2p_for_same_local_rank(
    offset,
    local_world_size,
    M_per_rank,
    N,
    input,
    output,
):

    rank = dl.rank()
    world_size = dl.num_ranks()
    node_id = rank // local_world_size
    nnodes = world_size // local_world_size
    local_rank = rank % local_world_size
    nelem_per_rank = M_per_rank * N

    remote_node_id = (offset + 1 + node_id) % nnodes
    remote_rank = local_rank + remote_node_id * local_world_size
    elem_size = tl.constexpr(input.dtype.element_ty.primitive_bitwidth) // 8
    libshmem_device.putmem_block(
        output + node_id * nelem_per_rank,
        input + remote_node_id * nelem_per_rank,
        nelem_per_rank * elem_size,
        remote_rank,
    )


def intra_node_scatter(
    input_intra_node,
    scatter_bufs_intra_node: List[torch.Tensor],
    scatter_signal_buf_intra_node: torch.Tensor,
    local_rank,
    stream,
    overlap_with_gemm=True,
):
    M, N = input_intra_node.shape
    local_world_size = len(scatter_bufs_intra_node)
    M_per_rank = M // local_world_size

    with torch.cuda.stream(stream):
        for i in range(0, local_world_size):

            remote_local_rank = (local_rank + i + 1) % local_world_size
            if overlap_with_gemm:

                wait_eq(
                    scatter_signal_buf_intra_node[remote_local_rank].data_ptr(),
                    1,
                    stream,
                    True,
                )
            remote_buf = scatter_bufs_intra_node[remote_local_rank][
                local_rank * M_per_rank : (local_rank + 1) * M_per_rank, :
            ]
            local_buf = input_intra_node[
                remote_local_rank * M_per_rank : (remote_local_rank + 1) * M_per_rank, :
            ]

            remote_buf.copy_(local_buf)


def reducer_scatter_for_each_node(input, stream, ctx: ReduceScatter2DContext):
    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    reduction_stream = ctx.reduction_stream
    num_reduction_sms = ctx.num_reduction_sms
    M, N = input.shape
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    rs_per_node_buf = ctx.rs_per_node_buf
    p2p_buf = ctx.p2p_buf

    with torch.cuda.stream(stream):
        for n in range(0, nnodes):

            cur_node_id = (node_id + n + 1) % nnodes
            input_intra_node = input[
                cur_node_id * M_per_node : (cur_node_id + 1) * M_per_node
            ]
            scatter_bufs_intra_node, scatter_signal_buf_intra_node = (
                ctx.get_scatter_bufs_and_signal_for_each_node(input, cur_node_id)
            )

            intra_node_scatter(
                input_intra_node,
                scatter_bufs_intra_node,
                scatter_signal_buf_intra_node,
                local_rank,
                stream,
                overlap_with_gemm=ctx.overlap_with_gemm,
            )

            rs_buf_cur_node = rs_per_node_buf[
                M_per_rank * cur_node_id : (cur_node_id + 1) * M_per_rank
            ]

            barrier_all_on_stream(ctx.barrier, stream)

            reduction_stream.wait_stream(stream)
            with torch.cuda.stream(reduction_stream):

                ring_reduce(
                    scatter_bufs_intra_node[local_rank],
                    rs_buf_cur_node,
                    local_rank,
                    local_world_size,
                    num_sms=-1 if n == nnodes - 1 else num_reduction_sms,
                )

                if nnodes > 1:
                    if n == nnodes - 1:
                        p2p_buf[
                            M_per_rank * node_id : M_per_rank * (node_id + 1)
                        ].copy_(
                            rs_per_node_buf[
                                M_per_rank * node_id : M_per_rank * (node_id + 1)
                            ]
                        )
                    else:
                        grid = lambda META: (ctx.num_p2p_sms,)
                        kernel_inter_node_p2p_for_same_local_rank[grid](
                            n,
                            local_world_size,
                            M_per_rank,
                            N,
                            rs_per_node_buf,
                            p2p_buf,
                            num_warps=16,
                        )

    stream.wait_stream(reduction_stream)
    if nnodes == 1:
        return rs_per_node_buf[: M_per_rank * nnodes]
    return p2p_buf[: M_per_rank * nnodes]


def reduce_scatter_multi_node(input, stream, ctx: ReduceScatter2DContext):

    M, N = input.shape
    M_per_rank = M // ctx.world_size
    ctx.p2p_stream.wait_stream(stream)

    rs_resutl_per_node = reducer_scatter_for_each_node(input, stream, ctx)
    barrier_all_on_stream(None, stream)
    output = torch.empty((M_per_rank, N), dtype=input.dtype, device=input.device)

    with torch.cuda.stream(stream):
        ring_reduce(rs_resutl_per_node, output, ctx.node_id, ctx.nnodes)
    return output


def reduce_scatter_2d_op(input, ctx: ReduceScatter2DContext):

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    reduction_stream = ctx.reduction_stream
    M, N = input.shape
    assert input.dtype == ctx.dtype
    assert ctx.max_M >= M and ctx.N == N
    assert M % ctx.world_size == 0

    current_stream = torch.cuda.current_stream()
    reduction_stream.wait_stream(current_stream)

    barrier_all_on_stream(None, current_stream)

    output = reduce_scatter_multi_node(input, current_stream, ctx)

    ctx.reset_barriers()
    return output


def torch_rs(
    input: torch.Tensor,
    TP_GROUP,
):
    M, N = input.shape
    rs_output = torch.empty(
        (M // WORLD_SIZE, N), dtype=input.dtype, device=input.device
    )
    torch.distributed.reduce_scatter_tensor(rs_output, input, group=TP_GROUP)
    return rs_output


if __name__ == "__main__":

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    TP_GROUP = triton_dist.utils.initialize_distributed()
    torch.cuda.synchronize()

    output_dtype = torch.bfloat16
    M, N = 8192, 16384
    rs_ctx = create_reduce_scater_2d_ctx(
        M, N, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, output_dtype, overlap_with_gemm=False
    )

    input = torch.rand((M, N), dtype=output_dtype).cuda()

    torch_output = torch_rs(input, TP_GROUP)

    pynvshmem.nvshmem_barrier_all()

    dist_triton_output = reduce_scatter_2d_op(input, rs_ctx)

    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    atol, rtol = 6e-2, 6e-2
    torch.testing.assert_close(torch_output, dist_triton_output, atol=atol, rtol=rtol)
    torch.cuda.synchronize()
    print(f"RANK {RANK}: pass!")
    torch.distributed.destroy_process_group()
