import datetime
import os

import torch

import triton
import triton_dist.language as dl
import triton.language as tl
from triton_dist import pynvshmem
from triton_dist.utils import dist_print, perf_func
from triton.language.extra.cuda.language_extra import atomic_cas, tid


@triton.jit
def kernel_ag_intra_node_nvlink_small_msg_split_msg(
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    symm_data_ptr,
    local_out_ptr,
    num_bytes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)

    num_bytes_per_pid = tl.cdiv(num_bytes, num_pids)
    for peer in range(0, num_ranks):
        ptr_in = dl.symm_at(symm_data_ptr, peer)
        ptr_out = local_out_ptr + num_bytes * peer
        offs = tl.arange(0, BLOCK_SIZE)

        for i in range(0, tl.cdiv(num_bytes_per_pid, BLOCK_SIZE)):
            mask = offs < num_bytes - i * BLOCK_SIZE - pid * num_bytes_per_pid
            data = tl.load(
                ptr_in + pid * num_bytes_per_pid + i * BLOCK_SIZE + offs, mask=mask
            )
            tl.store(
                ptr_out + pid * num_bytes_per_pid + i * BLOCK_SIZE + offs,
                data,
                mask=mask,
            )


@triton.jit
def kernel_ag_intra_node_nvlink_small_msg_split_rank(
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    symm_data_ptr,
    local_out_ptr,
    comm_buf_ptr,
    num_bytes,
    BLOCK_SIZE: tl.constexpr,
    need_sync: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)

    if need_sync:
        thread_idx = tid(0)

        if thread_idx < num_ranks:
            remote_ptr = dl.symm_at(comm_buf_ptr + pid * num_ranks + rank, thread_idx)
            while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
                pass
            while (
                atomic_cas(
                    comm_buf_ptr + pid * num_ranks + thread_idx, 1, 0, "sys", "acquire"
                )
                != 1
            ):
                pass

    num_ranks_per_pid = num_ranks // num_pids
    for local_peer in range(0, num_ranks_per_pid):
        peer = local_peer + pid * num_ranks_per_pid
        ptr_in = dl.symm_at(symm_data_ptr, peer)
        ptr_out = local_out_ptr + num_bytes * peer
        offs = tl.arange(0, BLOCK_SIZE)

        for i in range(0, tl.cdiv(num_bytes, BLOCK_SIZE)):
            mask = offs < num_bytes - i * BLOCK_SIZE
            data = tl.load(ptr_in + i * BLOCK_SIZE + offs, mask=mask)
            tl.store(ptr_out + i * BLOCK_SIZE + offs, data, mask=mask)


def nearest_power_two_u32(v):
    if v == 0:
        return 1
    v -= 1
    v = (v >> 1) | v
    v = (v >> 2) | v
    v = (v >> 4) | v
    v = (v >> 8) | v
    v = (v >> 16) | v
    v += 1
    return v


def ag_intra_node_nvlink_small_msg(
    symm_data, comm_buf, rank, num_ranks, need_block_level_sync=False
):
    symm_data = symm_data.to(torch.int8)
    msg_size_bytes = symm_data.shape[0]
    local_out = torch.empty(
        [num_ranks, msg_size_bytes], dtype=torch.int8, device="cuda"
    )
    grid = (8,)
    kernel_ag_intra_node_nvlink_small_msg_split_rank[grid](
        rank,
        num_ranks,
        symm_data,
        local_out,
        comm_buf,
        msg_size_bytes,
        nearest_power_two_u32(msg_size_bytes),
        need_block_level_sync,
    )
    return local_out


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(
        ranks=list(range(WORLD_SIZE)), backend="nccl"
    )
    torch.distributed.barrier()

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    use_block_sync = True

    msg_size_bytes = 1024 * 8
    symm_data = pynvshmem.nvshmem_create_tensor([msg_size_bytes], torch.int8)

    comm_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * WORLD_SIZE], torch.int32)
    comm_buf.fill_(0)
    symm_data.fill_(RANK)
    pynvshmem.nvshmemx_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    def func():
        if not use_block_sync:
            pynvshmem.nvshmemx_barrier_all_on_stream(current_stream.cuda_stream)
        results = ag_intra_node_nvlink_small_msg(
            symm_data, comm_buf, RANK, WORLD_SIZE, need_block_level_sync=use_block_sync
        )
        return results

    results = func()
    print(results)

    _, perf = perf_func(func, iters=100, warmup_iters=10)
    dist_print(
        f"rank{RANK}", perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE))
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for i in range(20):
            func()
    print(results)

    prof_dir = "prof/trace_ag_intra_node_nvlink_small_msg"
    os.makedirs(prof_dir, exist_ok=True)
    profiler.export_chrome_trace(f"{prof_dir}/rank{RANK}.json")

    torch.distributed.destroy_process_group()
