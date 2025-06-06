import torch
from triton_dist import pynvshmem
import os
import datetime

import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.utils import dist_print
from triton.language.extra.cuda.language_extra import __syncthreads


@triton.jit
def producer_consumer_kernel(
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    input_ptr,
    output_ptr,
    num_inputs: int,
    queue_ptr,
    signal_ptr,
    queue_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PRODUCER_SMS: tl.constexpr,
    NUM_CONSUMER_SMS: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid < NUM_PRODUCER_SMS:

        peer_rank = (rank + 1) % num_ranks
        offs = tl.arange(0, BLOCK_SIZE)
        for i in range(pid, num_inputs, NUM_PRODUCER_SMS):
            queue_offset = i % queue_size
            queue_repeat = i // queue_size
            token = dl.wait(
                dl.symm_at(signal_ptr, peer_rank) + queue_offset,
                1,
                "sys",
                "acquire",
                waitValue=queue_repeat * 2,
            )
            input_ptr = dl.consume_token(input_ptr, token)
            data = tl.load(input_ptr + i * BLOCK_SIZE + offs)

            tl.store(
                dl.symm_at(queue_ptr, peer_rank) + queue_offset * BLOCK_SIZE + offs,
                data,
            )

            __syncthreads()

            dl.notify(
                signal_ptr + queue_offset,
                peer_rank,
                signal=queue_repeat * 2 + 1,
                sig_op="set",
                comm_scope="intra_node",
            )
    elif pid < NUM_PRODUCER_SMS + NUM_CONSUMER_SMS:

        pid = pid - NUM_PRODUCER_SMS
        offs = tl.arange(0, BLOCK_SIZE)
        for i in range(pid, num_inputs, NUM_CONSUMER_SMS):
            queue_offset = i % queue_size
            queue_repeat = i // queue_size
            token = dl.wait(
                signal_ptr + queue_offset,
                1,
                "sys",
                "acquire",
                waitValue=queue_repeat * 2 + 1,
            )
            queue_ptr = dl.consume_token(queue_ptr, token)
            data = tl.load(queue_ptr + queue_offset * BLOCK_SIZE + offs)
            tl.store(output_ptr + i * BLOCK_SIZE + offs, data)
            __syncthreads()
            dl.notify(
                signal_ptr + queue_offset,
                rank,
                signal=queue_repeat * 2 + 2,
                sig_op="set",
                comm_scope="intra_node",
            )
    else:
        pass


def initialize_distributed():
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    assert WORLD_SIZE <= 8
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

    torch.cuda.synchronize()

    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
    return TP_GROUP


INPUT_SIZE = 2025
QUEUE_SIZE = 32
BLOCK_SIZE = 128


def main(TP_GROUP):
    stream = torch.cuda.current_stream()

    queue = pynvshmem.nvshmem_create_tensor((QUEUE_SIZE * BLOCK_SIZE,), torch.float32)
    signal = pynvshmem.nvshmem_create_tensor((QUEUE_SIZE,), torch.uint64)
    queue.fill_(-1)
    signal.fill_(0)

    pynvshmem.nvshmemx_barrier_all_on_stream(stream.cuda_stream)

    rank = TP_GROUP.rank()
    num_ranks = TP_GROUP.size()

    input_data = torch.randn((INPUT_SIZE * BLOCK_SIZE,), dtype=torch.float32).cuda()
    output_data = torch.empty_like(input_data)

    NUM_REPEAS = 20

    for iters in range(NUM_REPEAS):
        input_data = torch.randn((INPUT_SIZE * BLOCK_SIZE,), dtype=torch.float32).cuda()

        signal.fill_(0)
        pynvshmem.nvshmemx_barrier_all_on_stream(stream.cuda_stream)

        producer_consumer_kernel[(20,)](
            rank,
            num_ranks,
            input_data,
            output_data,
            INPUT_SIZE,
            queue,
            signal,
            QUEUE_SIZE,
            BLOCK_SIZE,
            16,
            4,
            num_warps=4,
        )

        inputs_all_ranks = [torch.empty_like(input_data) for _ in range(num_ranks)]
        torch.distributed.all_gather(inputs_all_ranks, input_data, group=TP_GROUP)
        golden = inputs_all_ranks[(rank - 1 + num_ranks) % num_ranks]
        if iters == NUM_REPEAS - 1:
            dist_print(
                f"rank{rank}",
                output_data,
                need_sync=True,
                allowed_ranks=list(range(num_ranks)),
            )
            dist_print(
                f"rank{rank}",
                golden,
                need_sync=True,
                allowed_ranks=list(range(num_ranks)),
            )
        assert torch.allclose(output_data, golden, atol=1e-5, rtol=1e-5)
        if iters == NUM_REPEAS - 1:
            dist_print(
                f"rank{rank} Passed✅!",
                need_sync=True,
                allowed_ranks=list(range(num_ranks)),
            )


TP_GROUP = initialize_distributed()

main(TP_GROUP)

torch.distributed.destroy_process_group()
