import datetime
import os
from dataclasses import dataclass

from triton_dist import pynvshmem
import torch

import triton
import triton.language as tl
from triton_dist.utils import perf_func
from triton.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import __syncthreads, tid


@dataclass
class AllGatherContext:
    rank: int
    node: int
    num_ranks: int
    num_nodes: int
    signal_tensor: torch.Tensor
    signal_value: int = 15
    max_buffer_size: int = 2 * 32 * 1024 * 1024


@triton.jit(do_not_specialize=["rank", "signal_value"])
def all_gather_push_1d_kernel(
    symm_ptr, bytes_per_rank, symm_flag, WORLD_SIZE: tl.constexpr, rank, signal_value
):
    pid = tl.program_id(0)
    thread_idx = tid(0)

    if pid == rank:
        peer = thread_idx
        if peer < WORLD_SIZE and peer != rank:

            libshmem_device.signal_wait_until(
                symm_flag + peer,
                libshmem_device.NVSHMEM_CMP_EQ,
                signal_value,
            )

        __syncthreads()
    else:
        peer = pid
        segment = rank

        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_value,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


def all_gather_push_1d(ctx: AllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_value += 1
    all_gather_push_1d_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.signal_tensor[ctx.signal_value % 2],
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_value,
    )
    return symm_buffer


@triton.jit(do_not_specialize=["rank", "signal_value"])
def all_gather_push_2d_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    NNODES: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    rank,
    signal_value,
):
    pid = tl.program_id(0)
    thread_idx = tid(0)

    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    node_id = rank // LOCAL_WORLD_SIZE
    local_rank = rank % LOCAL_WORLD_SIZE

    peer_rank = pid
    peer_node_id = peer_rank // LOCAL_WORLD_SIZE
    peer_local_rank = peer_rank % LOCAL_WORLD_SIZE
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:
        if peer_rank != rank:

            peer = peer_node_id * LOCAL_WORLD_SIZE + local_rank
            segment = rank
            libshmem_device.putmem_signal_nbi_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ptr + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_value,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )
        else:
            if thread_idx < WORLD_SIZE and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_value,
                )
            __syncthreads()
    else:
        peer = node_id * LOCAL_WORLD_SIZE + peer_local_rank
        segment = peer_node_id * LOCAL_WORLD_SIZE + local_rank

        if peer_node_id != node_id:
            if thread_idx == 0:
                libshmem_device.signal_wait_until(
                    symm_flag + segment,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_value,
                )
            __syncthreads()

        libshmem_device.putmem_signal_block(
            symm_ptr + segment * bytes_per_rank,
            symm_ptr + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_value,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


def all_gather_push_2d(ctx: AllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_value += 1
    all_gather_push_2d_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.signal_tensor[ctx.signal_value % 2],
        ctx.num_nodes,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_value,
        num_warps=32,
    )

    return symm_buffer


def perf_ag(func, ag_buffers: torch.Tensor, nbytes: int, ctx: AllGatherContext):
    nbytes_per_rank = nbytes // WORLD_SIZE

    ref_tensor = torch.arange(nbytes, dtype=torch.int8).cuda()
    ref_tensor = (
        torch.randint(0, 9999, [nbytes // 4], dtype=torch.int32).view(torch.int8).cuda()
    )
    torch.distributed.broadcast(ref_tensor, src=0)

    ag_buffer = ag_buffers[ctx.signal_value % 2]

    index_start, index_end = nbytes_per_rank * RANK, nbytes_per_rank * (RANK + 1)
    ag_buffer[index_start:index_end].copy_(ref_tensor[index_start:index_end])

    def _run_all_gather_triton():
        ag_buffer = ag_buffers[ctx.signal_value % 2][:nbytes]
        return func(ctx, ag_buffer)

    def _run_all_gather_nccl():
        torch.distributed.all_gather_into_tensor(
            ref_tensor, ref_tensor[index_start:index_end], group=TP_GROUP
        )

    result = _run_all_gather_triton()

    torch.testing.assert_close(result, ref_tensor, atol=0, rtol=0)
    print(f"✅ RANK[{RANK}] check passed")

    torch.cuda._sleep(1000000000)
    _, duration_per_iter_ms = perf_func(
        _run_all_gather_nccl,
        warmup_iters=5,
        iters=10,
    )
    gbps = nbytes * 1e-9 / (duration_per_iter_ms * 1e-3) * (WORLD_SIZE - 1)
    print(
        f"[NCCL] RANK = {RANK}, {nbytes // 1024} KB, Latency {duration_per_iter_ms * 1000:0.2f} us, Bus bandwith = {gbps:0.2f} GB/S"
    )

    pynvshmem.nvshmem_barrier_all()
    torch.cuda._sleep(1000000000)
    _, duration_per_iter_ms = perf_func(
        _run_all_gather_triton,
        warmup_iters=5,
        iters=10,
    )

    gbps = nbytes * 1e-9 / (duration_per_iter_ms * 1e-3) * (WORLD_SIZE - 1)
    print(
        f"[Triton] RANK = {RANK}, {nbytes // 1024} KB, Latency {duration_per_iter_ms * 1000:0.2f} us, Bus bandwith = {gbps:0.2f} GB/S"
    )


RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

torch.cuda.set_device(LOCAL_RANK)

torch.distributed.init_process_group(
    backend="nccl",
    world_size=WORLD_SIZE,
    rank=RANK,
    timeout=datetime.timedelta(seconds=1800),
)
assert torch.distributed.is_initialized()
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
torch.cuda.synchronize()


pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

nbytes = 8 * 1024


ag_buffer = pynvshmem.nvshmem_create_tensor((2, nbytes), torch.int8)
signals = [pynvshmem.nvshmem_create_tensor((1,), torch.uint64) for _ in range(2)]


ctx = AllGatherContext(
    rank=TP_GROUP.rank(),
    node=RANK // LOCAL_WORLD_SIZE,
    num_ranks=WORLD_SIZE,
    num_nodes=NNODES,
    signal_tensor=signals,
    signal_value=10,
)
print("using push 1d...")
perf_ag(
    all_gather_push_1d,
    ag_buffer,
    nbytes,
    ctx,
)
print("using push 2d...")
perf_ag(
    all_gather_push_2d,
    ag_buffer,
    nbytes,
    ctx,
)

del ag_buffer
del signals[-1]
del signals[-1]
torch.distributed.destroy_process_group()
