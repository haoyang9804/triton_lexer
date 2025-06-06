from hip import hip
import os
import triton
import torch
import torch.distributed as dist
from triton.language.extra import libdevice
from triton.language.extra.hip import libdevice
import time
import pyrocshmem
import random

GLOBAL_RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
torch.cuda.set_device(LOCAL_RANK)

dist.init_process_group(backend="nccl")


SIGNAL_DTYPE = torch.uint64
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def wait_eq_driver_impl(ptr: int, val: int, stream: torch.cuda.Stream):
    mask = 0xFFFFFFFF
    if SIGNAL_DTYPE == torch.int32:
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
    hip_check(call_result)


def notify_driver_impl(ptr: int, val: int, stream: torch.cuda.Stream):
    if SIGNAL_DTYPE == torch.int32:
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
    hip_check(call_result)


@triton.jit
def kernel_notify(signal, val):

    libdevice.store_release_system(signal)


def wait_eq(signal, val, stream, use_driver_api=True):
    if use_driver_api:
        wait_eq_driver_impl(signal.data_ptr(), val, stream)
    else:

        raise NotImplementedError()


def notify(signal, val, stream, use_driver_api=False):
    if use_driver_api:
        notify_driver_impl(signal.data_ptr(), val, stream)
    else:
        one = torch.ones(
            [
                1,
            ],
            dtype=torch.int32,
        ).cuda()
        kernel_notify[1,](signal, one)


def wait_and_reduce(signal, ag_buffer):
    M, N = ag_buffer.shape
    local_M = M // TP_GROUP.size()
    sum = torch.zeros((local_M, N), dtype=ag_buffer.dtype, device=ag_buffer.device)
    current_stream = torch.cuda.current_stream()
    for i in range(WORLD_SIZE):

        wait_eq(signal[i], 1, current_stream)

        sum = sum + ag_buffer[i * local_M : (i + 1) * local_M, :]
    return sum


def send_and_notify(input, ag_buffers, signals):
    local_M, N = input.shape
    current_stream = torch.cuda.current_stream()
    nbytes = input.numel() * input.element_size()
    for i in range(WORLD_SIZE):
        remote_rank = (i + LOCAL_RANK) % WORLD_SIZE

        dst_tensor = ag_buffers[remote_rank][
            LOCAL_RANK * local_M : (LOCAL_RANK + 1) * local_M, :
        ]
        call_result = hip.hipMemcpyAsync(
            dst_tensor.data_ptr(),
            input.data_ptr(),
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
            current_stream.cuda_stream,
        )
        hip_check(call_result)

        notify(signals[remote_rank][LOCAL_RANK], 1, current_stream)


def dist_ag_and_reduce(input, ag_buffers, signals, notify_stream):
    output = wait_and_reduce(signals[LOCAL_RANK], ag_buffers[LOCAL_RANK])

    time.sleep(random.randint(1, 5))
    with torch.cuda.stream(notify_stream):
        send_and_notify(input, ag_buffers, signals)
    return output


def torch_ag_and_reduce(input):
    local_M, N = input.shape
    M = local_M * TP_GROUP.size()
    full_input = torch.zeros(
        (M, input.size(1)), dtype=input.dtype, device=input.device, requires_grad=False
    )
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
    sum = torch.zeros((local_M, N), dtype=input.dtype, device=input.device)
    for i in range(WORLD_SIZE):
        sum = sum + full_input[i * local_M : (i + 1) * local_M, :]
    return sum


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


if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    torch.cuda.synchronize()
    dtype = torch.float16
    M = 16384
    N = 12288
    local_M = M // TP_GROUP.size()
    input = (
        torch.rand((local_M, N), dtype=dtype).cuda()
        / 100
        * ((TP_GROUP.rank() + 1) ** 2)
    )
    ag_buffers = create_tensor_ipc([M, N], dtype)
    signals = create_tensor_ipc(
        [
            WORLD_SIZE,
        ],
        SIGNAL_DTYPE,
    )
    signals[LOCAL_RANK].fill_(0)
    torch.cuda.synchronize()
    dist.barrier()
    notify_stream = torch.cuda.Stream(priority=-1)
    dist.barrier()

    dist_output = dist_ag_and_reduce(input, ag_buffers, signals, notify_stream)
    dist.barrier()

    gloden_output = torch_ag_and_reduce(input)
    torch.cuda.synchronize()
    dist.barrier()

    torch.testing.assert_close(gloden_output, dist_output, rtol=0, atol=0)
    print(f"RANK[{LOCAL_RANK}]: Test Passed!")

    dist.barrier()
    dist.destroy_process_group()
