from hip import hip
import os
import triton
import torch
import torch.distributed as dist
import time
import pyrocshmem
import triton.language as tl
from triton.language.extra.hip import libdevice

from contextlib import nullcontext

GLOBAL_RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
torch.cuda.set_device(LOCAL_RANK)

dist.init_process_group(backend="nccl")

TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def get_torch_prof_ctx(do_prof: bool):
    ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        )
        if do_prof
        else nullcontext()
    )
    return ctx


def create_tensor_ipc(shape, dtype):
    input_buffer = torch.zeros(
        shape, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
    )
    input_buffer_offset = input_buffer.storage_offset()
    shm_handle = input_buffer._typed_storage()._share_cuda_()[1]
    shm_handle = shm_handle[2:]
    shm_offset = input_buffer._typed_storage()._share_cuda_()[3]
    shm_handle_ts_cuda = torch.ByteTensor(
        torch.ByteStorage._from_buffer(shm_handle)
    ).cuda()
    shm_handles = [torch.empty_like(shm_handle_ts_cuda) for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather(shm_handles, shm_handle_ts_cuda, group=TP_GROUP)
    offset_value = shm_offset + input_buffer_offset
    offset_list = [None for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather_object(offset_list, offset_value, group=TP_GROUP)
    shm_buffers = pyrocshmem.rocshmem_get_tensors_from_ipchandle(
        LOCAL_RANK, WORLD_SIZE, shm_handles, offset_list, input_buffer.shape, dtype
    )
    shm_buffers.insert(LOCAL_RANK, input_buffer)
    return shm_buffers


@triton.jit
def barrier_all(rank, num_ranks, comm_buf_base_ptrs):
    tid = libdevice.thread_idx(axis=0)
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while (
            tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0
        ):
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(
            tl.pointer_type(tl.int32)
        )
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


@triton.jit
def barrier_all2(rank, num_ranks, comm_buf_base_ptrs, zero_ptr, one_ptr):
    tid = libdevice.thread_idx(axis=0)
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while (
            libdevice.atom_cas_release_relaxed_system(
                remote_base_ptr + rank, zero_ptr, one_ptr
            )
            == 0
        ):
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(
            tl.pointer_type(tl.int32)
        )
        while (
            libdevice.atom_cas_acquire_relaxed_system(
                local_base_ptr + i, one_ptr, zero_ptr
            )
            == 0
        ):
            pass

    tl.debug_barrier()


if __name__ == "__main__":
    profile = True
    torch.cuda.set_device(LOCAL_RANK)
    torch.cuda.synchronize()
    dtype = torch.float16
    signals = create_tensor_ipc(
        [
            WORLD_SIZE,
        ],
        torch.int32,
    )
    if LOCAL_RANK % 4 == 0:
        print(f"dbg signals: {signals}")
    signals[LOCAL_RANK].fill_(0)
    torch.cuda.synchronize()
    dist.barrier()
    ctx = get_torch_prof_ctx(profile)
    zero = torch.zeros(
        [
            8,
        ],
        dtype=torch.int32,
    ).cuda()
    one = torch.ones(
        [
            8,
        ],
        dtype=torch.int32,
    ).cuda()

    signals_ptr = torch.tensor([t.data_ptr() for t in signals]).cuda()
    dist.barrier()
    with ctx:
        dist.barrier()
        time.sleep(LOCAL_RANK)
        barrier_all[(1,)](LOCAL_RANK, LOCAL_WORLD_SIZE, signals_ptr)

    if profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    torch.cuda.synchronize()
    dist.barrier()
    print("after sync:", signals[LOCAL_RANK])

    dist.destroy_process_group()
