import torch
import triton
import triton.language as tl
from triton.language.extra import libshmem_device
from triton_dist import pynvshmem

from typing import List
from cuda import cuda
from triton_dist.utils import CUDA_CHECK

from triton_dist.utils import initialize_distributed, dist_print


def cp_engine_producer_all_gather_full_mesh_pull(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    ag_stream: torch.cuda.Stream,
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape

    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    with torch.cuda.stream(ag_stream):
        for src_rank in rank_orders:
            if src_rank == rank:
                continue

            dst = remote_tensor_buffers[rank][
                src_rank * M_per_rank : (src_rank + 1) * M_per_rank, :
            ]
            src = remote_tensor_buffers[src_rank][
                src_rank * M_per_rank : (src_rank + 1) * M_per_rank, :
            ]
            dst.copy_(src)
            (err,) = cuda.cuStreamWriteValue32(
                ag_stream.cuda_stream,
                barrier_buffers[rank][src_rank].data_ptr(),
                1,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            CUDA_CHECK(err)


@triton.jit
def nvshmem_device_producer_all_gather_2d_put_block_kernel(
    remote_tensor_ptr,
    signal_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    signal_target,
    local_rank,
    world_size,
    DISPATCH_BLOCK_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    if pid < DISPATCH_BLOCK_NUM:
        peer = (local_rank + pid + 1) % world_size
        segment = local_rank
        libshmem_device.putmem_signal_block(
            remote_tensor_ptr + segment * elem_per_rank,
            remote_tensor_ptr + segment * elem_per_rank,
            elem_per_rank * size_per_elem,
            signal_buffer_ptr + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()
    num_ranks = TP_GROUP.size()
    assert num_ranks == 8, "This tutorial is designed for intra-node"

    M = 8192
    N = 12288
    M_per_rank = M // num_ranks
    dtype = torch.float16
    signal_dtype = torch.uint64

    local_data = torch.randn([M_per_rank, N], dtype=dtype, device="cuda")
    ag_buffer_ptrs = pynvshmem.nvshmem_create_tensor_list_intra_node([M, N], dtype)
    signal = pynvshmem.nvshmem_create_tensor_list_intra_node(
        ([num_ranks]), signal_dtype
    )

    golden = torch.empty([M, N], dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(golden, local_data, group=TP_GROUP)

    ag_buffer_ptrs[rank].fill_(-1)
    ag_buffer_ptrs[rank][rank * M_per_rank : (rank + 1) * M_per_rank,].copy_(local_data)
    signal[rank].fill_(0)

    pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    cp_engine_producer_all_gather_full_mesh_pull(
        rank, num_ranks, local_data, ag_buffer_ptrs, torch.cuda.current_stream(), signal
    )

    dist_print(
        f"Rank {rank} CpEngine Result:\n",
        ag_buffer_ptrs[rank],
        need_sync=True,
        allowed_ranks="all",
    )
    dist_print(
        f"Rank {rank} CpEngine Signal:\n",
        signal[rank],
        need_sync=True,
        allowed_ranks="all",
    )
    assert torch.allclose(golden, ag_buffer_ptrs[rank], atol=1e-5, rtol=1e-5)
    dist_print(f"Rank {rank}", "Pass!✅", need_sync=True, allowed_ranks="all")

    ag_buffer_ptrs[rank].fill_(-1)
    ag_buffer_ptrs[rank][rank * M_per_rank : (rank + 1) * M_per_rank,].copy_(local_data)
    signal[rank].fill_(0)

    pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    grid = lambda META: (int(num_ranks),)
    nvshmem_device_producer_all_gather_2d_put_block_kernel[grid](
        ag_buffer_ptrs[rank],
        signal[rank],
        M_per_rank * N,
        local_data.element_size(),
        1,
        rank,
        num_ranks,
        num_ranks,
    )

    pynvshmem.nvshmem_barrier_all()

    dist_print(
        f"Rank {rank} NVSHMEM Result:\n",
        ag_buffer_ptrs[rank],
        need_sync=True,
        allowed_ranks="all",
    )
    dist_print(
        f"Rank {rank} NVSHMEM Signal:\n",
        signal[rank],
        need_sync=True,
        allowed_ranks="all",
    )
    assert torch.allclose(golden, ag_buffer_ptrs[rank], atol=1e-5, rtol=1e-5)
    dist_print(f"Rank {rank}", "Pass!✅", need_sync=True, allowed_ranks="all")

    del ag_buffer_ptrs
    del signal

    torch.distributed.destroy_process_group()
