import torch
import triton
from triton_dist import pynvshmem
import triton_dist.language as dl

import os
import datetime


@triton.jit
def test_notify_set(ptr):
    mype = dl.rank()
    npes = dl.num_ranks()
    peer = (mype + 1) % npes
    dl.notify(ptr, peer, signal=mype, sig_op="set", comm_scope="inter_node")


@triton.jit
def test_notify_add(ptr):
    dl.notify(ptr, 0, signal=1, sig_op="add", comm_scope="intra_node")


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

    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    t = pynvshmem.nvshmem_create_tensor((8,), torch.uint64)
    t.fill_(0)
    pynvshmem.nvshmem_barrier_all()
    test_notify_set[(1,)](t)
    pynvshmem.nvshmem_barrier_all()

    assert t[0].item() == (RANK + WORLD_SIZE - 1) % WORLD_SIZE

    t.fill_(0)
    pynvshmem.nvshmem_barrier_all()
    test_notify_add[(1,)](t)
    pynvshmem.nvshmem_barrier_all()
    ref = WORLD_SIZE if RANK == 0 else 0
    assert t[0].item() == ref

    print(f"RANK {RANK}: pass.")
    torch.distributed.destroy_process_group()
