import triton
import torch
import triton.language as tl

from triton_dist.utils import (
    HIP_CHECK,
)
from hip import hip

from triton.language.extra.hip.libdevice import (
    thread_idx,
    load_acquire_system,
)


@triton.jit
def wait_eq_sys(barrier_ptr, value):
    tid = thread_idx(axis=0)
    if tid == 0:
        while load_acquire_system(barrier_ptr) != value:
            pass

    tl.debug_barrier()


@triton.jit
def barrier_all_ipc(rank, num_ranks, comm_buf_base_ptrs):
    tid = thread_idx(axis=0)
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


def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_bufs_ptr,
    stream,
):
    with torch.cuda.stream(stream):
        barrier_all_ipc[(1,)](rank, num_ranks, sync_bufs_ptr)


def wait_eq(ptr: int, val: int, stream: torch.cuda.Stream, require_i64=False):
    mask = 0xFFFFFFFF
    if not require_i64:
        call_result = hip.hipStreamWaitValue32(
            stream.cuda_stream,
            ptr,
            val,
            hip.hipStreamWaitValueEq,
            mask,
        )
    else:
        call_result = hip.hipStreamWaitValue64(
            stream.cuda_stream,
            ptr,
            val,
            hip.hipStreamWaitValueEq,
            mask,
        )
    HIP_CHECK(call_result)


def set_signal(ptr: int, val: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        call_result = hip.hipStreamWriteValue32(
            stream.cuda_stream,
            ptr,
            val,
            0,
        )
    else:
        call_result = hip.hipStreamWriteValue64(
            stream.cuda_stream,
            ptr,
            val,
            0,
        )
    HIP_CHECK(call_result)
