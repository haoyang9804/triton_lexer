from triton_dist import pynvshmem
import torch
import torch.distributed
from dataclasses import dataclass
from typing import List

import triton
import triton.language as tl
from triton.language.extra import libshmem_device

from triton.language.extra.cuda.language_extra import (
    __syncthreads,
    tid,
    ntid,
    load_v4_u32,
    load_v2_b64,
    store_v2_u32,
    st,
    multimem_st_b64,
    multimem_st_v2_b32,
)


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_pull_kernel(
    symm_ptr, bytes_per_rank, symm_flag, world_size, rank, signal_target
):
    pid = tl.program_id(0)
    thread_idx = tid(0)
    if pid == rank:
        if thread_idx != rank and thread_idx < world_size:
            libshmem_device.signal_op(
                symm_flag + rank,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                thread_idx,
            )
        __syncthreads()
    else:
        peer = pid
        if thread_idx == 0:
            libshmem_device.signal_wait_until(
                symm_flag + peer, libshmem_device.NVSHMEM_CMP_EQ, signal_target
            )
        __syncthreads()
        libshmem_device.getmem_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + peer * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + peer * bytes_per_rank,
            bytes_per_rank,
            peer,
        )


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_numa_2d_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    n_numa_nodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):
    tl.static_assert(n_numa_nodes == 2, "only support NUMA node == 2")
    numa_world_size = world_size // n_numa_nodes
    local_rank = rank % numa_world_size
    nid = rank // numa_world_size

    pid = tl.program_id(0)
    peer_nid = pid // numa_world_size
    peer_local_rank = pid % numa_world_size
    thread_idx = tid(0)

    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:
        if peer_nid != nid:
            peer_to = peer_nid * numa_world_size + local_rank
            libshmem_device.putmem_signal_block(
                symm_ptr + rank * bytes_per_rank,
                symm_ptr + rank * bytes_per_rank,
                bytes_per_rank,
                symm_flag + rank,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer_to,
            )
        else:

            if thread_idx < world_size and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()
    else:
        peer = nid * numa_world_size + peer_local_rank
        segment = peer_nid * numa_world_size + local_rank
        if peer_nid != nid:
            if thread_idx == 0:
                libshmem_device.signal_wait_until(
                    symm_flag + segment, libshmem_device.NVSHMEM_CMP_EQ, signal_target
                )
            __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_numa_2d_ll_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    symm_ll_buffer,
    n_numa_nodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):
    tl.static_assert(n_numa_nodes == 2, "only support NUMA node == 2")
    numa_world_size = world_size // n_numa_nodes
    local_rank = rank % numa_world_size
    nid = rank // numa_world_size

    pid = tl.program_id(0)
    peer_nid = pid // numa_world_size
    peer_local_rank = pid % numa_world_size
    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4

    symm_ll_buffer = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:
        if peer_nid != nid:
            segment = peer_nid * numa_world_size + local_rank
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ll_buffer + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )
            __syncthreads()
            if thread_idx == 0:
                st(symm_flag + segment, signal_target, scope="gpu", semantic="release")
        else:
            _pack_ll_block(
                symm_ll_buffer + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )
            __syncthreads()

            peer_to_nid = 1 - nid
            peer_to = peer_to_nid * numa_world_size + local_rank
            libshmem_device.putmem_block(
                symm_ll_buffer + rank * bytes_per_rank * 2,
                symm_ll_buffer + rank * bytes_per_rank * 2,
                bytes_per_rank * 2,
                peer_to,
            )

            if thread_idx < world_size and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()

    else:
        peer = nid * numa_world_size + peer_local_rank
        segment = peer_nid * numa_world_size + local_rank
        if peer_nid != nid:
            if thread_idx == 0:
                libshmem_device.signal_wait_until(
                    symm_flag + segment, libshmem_device.NVSHMEM_CMP_EQ, signal_target
                )
            __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_numa_2d_ll_multinode_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    symm_ll_buffer,
    nnodes: tl.constexpr,
    n_numa_nodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):

    tl.static_assert(n_numa_nodes == 2, "only support NUMA node == 2")
    local_world_size = world_size // nnodes
    node_id = rank // local_world_size
    local_rank = rank % local_world_size
    numa_world_size = local_world_size // n_numa_nodes
    numa_rank = local_rank % numa_world_size
    local_numa_id = local_rank // numa_world_size
    global_numa_id = rank // numa_world_size

    pid = tl.program_id(0)
    peer_node_id = pid // local_world_size
    peer_local_rank = pid % local_world_size
    peer_numa_rank = peer_local_rank % numa_world_size
    peer_local_numa_id = peer_local_rank // numa_world_size
    peer_global_numa_id = pid // numa_world_size
    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4

    symm_ll_buffer = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    is_intra_numa = numa_rank != peer_numa_rank
    is_inter_numa = node_id == peer_node_id and (
        local_numa_id != peer_local_numa_id and numa_rank == peer_numa_rank
    )

    if is_intra_numa and global_numa_id == peer_global_numa_id:
        peer = global_numa_id * numa_world_size + peer_numa_rank
        segment = rank
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )
    elif is_intra_numa and global_numa_id != peer_global_numa_id:
        peer = global_numa_id * numa_world_size + peer_numa_rank
        segment = (
            peer_node_id * local_world_size
            + peer_local_numa_id * numa_world_size
            + numa_rank
        )

        if thread_idx == 0:
            libshmem_device.signal_wait_until(
                symm_flag + segment, libshmem_device.NVSHMEM_CMP_EQ, signal_target
            )
        __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )
    elif is_inter_numa:
        peer = (
            node_id * local_world_size
            + peer_local_numa_id * numa_world_size
            + peer_numa_rank
        )
        segment = rank
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )
    else:
        if peer_node_id != node_id:
            segment = peer_global_numa_id * numa_world_size + peer_numa_rank
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ll_buffer + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )
            __syncthreads()

            if thread_idx == 0:
                st(symm_flag + segment, signal_target, scope="gpu", semantic="release")
        else:
            _pack_ll_block(
                symm_ll_buffer + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )
            __syncthreads()

            for i in range(world_size // numa_world_size):
                if i // n_numa_nodes != node_id:
                    peer_to = numa_rank + i * numa_world_size
                    libshmem_device.putmem_nbi_warp(
                        symm_ll_buffer + rank * bytes_per_rank * 2,
                        symm_ll_buffer + rank * bytes_per_rank * 2,
                        bytes_per_rank * 2,
                        peer_to,
                    )

            if thread_idx < world_size and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_2d_kernel(
    symm_ptr, bytes_per_rank, symm_flag, NNODES, WORLD_SIZE, rank, signal_target
):
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    local_rank = rank % LOCAL_WORLD_SIZE
    node_id = rank // LOCAL_WORLD_SIZE
    rank_base = node_id * LOCAL_WORLD_SIZE

    pid = tl.program_id(0)
    peer_rank = pid
    peer_node_id = peer_rank // LOCAL_WORLD_SIZE
    peer_local_rank = peer_rank % LOCAL_WORLD_SIZE
    thread_idx = tid(0)
    if peer_local_rank == local_rank:
        if peer_rank != rank:
            peer = peer_node_id * LOCAL_WORLD_SIZE + local_rank
            segment = rank
            libshmem_device.putmem_signal_nbi_block(
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )
        else:
            if thread_idx < WORLD_SIZE and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()
    else:
        peer = rank_base + peer_local_rank
        segment = peer_node_id * LOCAL_WORLD_SIZE + local_rank
        if peer_node_id != node_id:
            if thread_idx == 0:
                libshmem_device.signal_wait_until(
                    symm_flag + segment,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_3d_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_ll_buffer,
    symm_flag,
    NNODES,
    N_NUMA_NODES,
    WORLD_SIZE,
    rank,
    signal_target,
    INTER_NODE_WITH_LL: tl.constexpr = False,
):

    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    NUMA_WORLD_SIZE = LOCAL_WORLD_SIZE // N_NUMA_NODES
    local_rank = rank % LOCAL_WORLD_SIZE
    node_id = rank // LOCAL_WORLD_SIZE
    numa_rank = local_rank % NUMA_WORLD_SIZE
    local_numa_id = local_rank // NUMA_WORLD_SIZE

    pid = tl.program_id(0)
    peer_rank = pid
    peer_node_id = peer_rank // LOCAL_WORLD_SIZE
    peer_local_rank = peer_rank % LOCAL_WORLD_SIZE
    peer_numa_rank = peer_local_rank % NUMA_WORLD_SIZE
    peer_local_numa_id = peer_local_rank // NUMA_WORLD_SIZE

    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))
    symm_ll_buffer = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    if peer_local_rank == local_rank:
        if peer_node_id != node_id:
            if INTER_NODE_WITH_LL:
                segment = peer_node_id * LOCAL_WORLD_SIZE + local_rank
                _recv_ll_block(
                    symm_ptr + segment * bytes_per_rank,
                    symm_ll_buffer + segment * bytes_per_rank * 2,
                    num_ints,
                    signal_target,
                )
                __syncthreads()
                if thread_idx == 0:
                    st(
                        symm_flag + segment,
                        signal_target,
                        scope="gpu",
                        semantic="release",
                    )
        else:
            wid = thread_idx // 32
            if INTER_NODE_WITH_LL:
                segment = rank
                _pack_ll_block(
                    symm_ll_buffer + rank * bytes_per_rank * 2,
                    symm_ptr + rank * bytes_per_rank,
                    num_ints,
                    signal_target,
                    2048,
                )
                __syncthreads()

                if wid < NNODES and wid != node_id:
                    peer = wid * LOCAL_WORLD_SIZE + local_rank
                    libshmem_device.putmem_nbi_warp(
                        symm_ll_buffer + segment * bytes_per_rank * 2,
                        symm_ll_buffer + segment * bytes_per_rank * 2,
                        bytes_per_rank * 2,
                        peer,
                    )
            else:
                if wid < NNODES and wid != node_id:
                    peer = wid * LOCAL_WORLD_SIZE + local_rank
                    segment = rank
                    libshmem_device.putmem_signal_nbi_warp(
                        symm_ptr + segment * bytes_per_rank,
                        symm_ptr + segment * bytes_per_rank,
                        bytes_per_rank,
                        symm_flag + segment,
                        signal_target,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        peer,
                    )

            __syncthreads()
            if thread_idx < WORLD_SIZE and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()
    else:

        if NNODES > 1:
            if thread_idx < WORLD_SIZE and (
                thread_idx % LOCAL_WORLD_SIZE == local_rank and thread_idx != rank
            ):
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()

        if peer_numa_rank == numa_rank:
            peer = (
                node_id * LOCAL_WORLD_SIZE
                + peer_local_numa_id * NUMA_WORLD_SIZE
                + numa_rank
            )
            segment = peer_node_id * LOCAL_WORLD_SIZE + local_rank
            libshmem_device.putmem_signal_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ptr + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )
        else:
            peer = (
                node_id * LOCAL_WORLD_SIZE
                + local_numa_id * NUMA_WORLD_SIZE
                + peer_numa_rank
            )
            segment = (
                peer_node_id * LOCAL_WORLD_SIZE
                + peer_local_numa_id * NUMA_WORLD_SIZE
                + numa_rank
            )

            if peer_local_numa_id != local_numa_id:
                if thread_idx == 0:
                    libshmem_device.signal_wait_until(
                        symm_flag + segment,
                        libshmem_device.NVSHMEM_CMP_EQ,
                        signal_target,
                    )
                __syncthreads()

            libshmem_device.putmem_signal_block(
                symm_ptr + segment * bytes_per_rank,
                symm_ptr + segment * bytes_per_rank,
                bytes_per_rank,
                symm_flag + segment,
                signal_target,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                peer,
            )


@triton.jit
def _recv_ll_block(dest_ptr, src_ptr, num_ints, ll_flag):

    thread_idx = tid(0)
    block_size = ntid(0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))

    for n in range(thread_idx, num_ints // 2, block_size):
        data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        while flag1 != ll_flag or flag2 != ll_flag:
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        store_v2_u32(dest_ptr + n * 2, data1, data2)


@triton.jit(do_not_specialize=["ll_flag"])
def _pack_ll_block(dest_ptr, src_ptr, num_ints, ll_flag, BLOCK_SIZE: tl.constexpr):

    iters = tl.cdiv(num_ints, BLOCK_SIZE)
    src_ptr = tl.cast(src_ptr, dtype=tl.pi32_t)
    dest_ptr = tl.cast(dest_ptr, dtype=tl.pi32_t)
    for n in range(iters):
        src_offsets = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        src_mask = src_offsets < num_ints
        src = tl.load(src_ptr + src_offsets, mask=src_mask)
        flags = tl.full((BLOCK_SIZE,), ll_flag, tl.int32)
        dst = tl.interleave(src, flags)
        dest_offset = n * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
        dest_mask = dest_offset < num_ints * 2
        tl.store(dest_ptr + dest_offset, dst, mask=dest_mask)


@triton.jit
def _recv_ll_and_multimem_st_block(dest_ptr, src_ptr, num_ints, ll_flag):

    thread_idx = tid(0)
    block_size = ntid(0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))
    dest_mc_ptr = libshmem_device.remote_mc_ptr(
        libshmem_device.NVSHMEMX_TEAM_NODE, dest_ptr
    )

    for n in range(thread_idx, num_ints // 2, block_size):
        data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        while flag1 != ll_flag or flag2 != ll_flag:
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        multimem_st_v2_b32(dest_mc_ptr + n * 2, data1, data2)


@triton.jit(do_not_specialize=["ll_flag"])
def _recv_ll_and_multimem_st_ll_block(dest_ptr, src_ptr, num_ints, ll_flag):

    thread_idx = tid(0)
    block_size = ntid(0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))
    dest_mc_ptr = libshmem_device.remote_mc_ptr(
        libshmem_device.NVSHMEMX_TEAM_NODE, dest_ptr
    )

    for n in range(thread_idx, num_ints // 2, block_size):
        data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        while flag1 != ll_flag or flag2 != ll_flag:
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
        multimem_st_v2_b32(dest_mc_ptr + n * 4, data1, flag1)
        multimem_st_v2_b32(dest_mc_ptr + n * 4 + 2, data2, flag2)


@triton.jit
def broadcast_naive_block(dst_ptr, src_ptr, nbytes):
    thread_idx = tid(axis=0)
    block_dim = ntid(axis=0)
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int8))
    dst_ptr = tl.cast(dst_ptr, tl.pointer_type(tl.int8))
    dst_mc_ptr = libshmem_device.remote_mc_ptr(
        libshmem_device.NVSHMEMX_TEAM_NODE, dst_ptr
    )
    num_int4 = nbytes // 16
    for n in range(thread_idx, num_int4, block_dim):
        val0, val1 = load_v2_b64(src_ptr + 16 * n)
        multimem_st_b64(dst_mc_ptr + n * 16, val0)
        multimem_st_b64(dst_mc_ptr + n * 16 + 8, val1)


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_2d_ll_multimem_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_ll_buffer,
    nnodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):

    local_world_size = world_size // nnodes
    local_rank = rank % local_world_size
    nid = rank // local_world_size

    pid = tl.program_id(0)
    peer_nid = pid // local_world_size
    peer_local_rank = pid % local_world_size
    num_ints = bytes_per_rank // 4
    thread_idx = tid(axis=0)

    ll_buffer_int8 = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:
        if nid != peer_nid:
            segment = peer_nid * local_world_size + local_rank
            _recv_ll_and_multimem_st_ll_block(
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )
        else:
            _pack_ll_block(
                ll_buffer_int8 + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )
            __syncthreads()
            wid = thread_idx // 32

            if wid < nnodes and wid != nid:
                peer_to = wid * local_world_size + local_rank
                libshmem_device.putmem_nbi_warp(
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    bytes_per_rank * 2,
                    peer_to,
                )

            segment = peer_nid * local_world_size + local_rank
            broadcast_naive_block(
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                bytes_per_rank * 2,
            )
    else:
        segment_recv_local = peer_nid * local_world_size + peer_local_rank
        _recv_ll_block(
            symm_ptr + segment_recv_local * bytes_per_rank,
            ll_buffer_int8 + segment_recv_local * bytes_per_rank * 2,
            num_ints,
            signal_target,
        )


@triton.jit(do_not_specialize=["rank", "signal_target"])
def _forward_push_2d_ll_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_flag,
    symm_ll_buffer,
    nnodes: tl.constexpr,
    world_size: tl.constexpr,
    rank,
    signal_target,
):
    local_world_size = world_size // nnodes
    local_rank = rank % local_world_size
    nid = rank // local_world_size

    pid = tl.program_id(0)
    peer_nid = pid // local_world_size
    peer_local_rank = pid % local_world_size
    thread_idx = tid(0)
    num_ints = bytes_per_rank // 4

    ll_buffer_int8 = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:
        if peer_nid != nid:
            segment = peer_nid * local_world_size + local_rank
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )
            __syncthreads()
            if thread_idx == 0:
                st(symm_flag + segment, signal_target, scope="gpu", semantic="release")
        else:
            _pack_ll_block(
                ll_buffer_int8 + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )
            __syncthreads()
            wid = thread_idx // 32
            if wid < nnodes and wid != nid:
                peer_to = wid * local_world_size + local_rank
                libshmem_device.putmem_nbi_warp(
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    bytes_per_rank * 2,
                    peer_to,
                )

            if thread_idx < world_size and thread_idx != rank:
                libshmem_device.signal_wait_until(
                    symm_flag + thread_idx,
                    libshmem_device.NVSHMEM_CMP_EQ,
                    signal_target,
                )
            __syncthreads()

    else:
        peer = nid * local_world_size + peer_local_rank
        segment = peer_nid * local_world_size + local_rank
        if peer_nid != nid:
            if thread_idx == 0:
                libshmem_device.signal_wait_until(
                    symm_flag + segment, libshmem_device.NVSHMEM_CMP_EQ, signal_target
                )
            __syncthreads()
        libshmem_device.putmem_signal_block(
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            tl.cast(symm_ptr, tl.pointer_type(tl.int8)) + segment * bytes_per_rank,
            bytes_per_rank,
            symm_flag + segment,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


@dataclass
class FastAllGatherContext:
    rank: int
    node: int
    num_ranks: int
    num_nodes: int
    signal_tensor: torch.Tensor
    ll_buffers: List[torch.Tensor]
    grid_barrier: torch.Tensor
    max_buffer_size: int = 2 * 32 * 1024 * 1024
    signal_target: int = 15

    def update(self, rank, node, num_ranks, num_nodes, signal_target):
        self.rank = rank
        self.node = node
        self.num_ranks = num_ranks
        self.num_nodes = num_nodes
        self.signal_target = signal_target


def create_fast_allgather_context(
    rank, node, num_ranks, num_nodes, max_buffer_size: int = 2 * 32 * 1024 * 1024
):
    signal_tensor = pynvshmem.nvshmem_create_tensor((num_ranks,), torch.uint64)
    signal_tensor.zero_()
    ll_buffers = [
        pynvshmem.nvshmem_create_tensor((max_buffer_size,), torch.int8)
        for _ in range(2)
    ]
    grid_barrier = torch.zeros((1,), dtype=torch.uint32, device="cuda")

    ctx = FastAllGatherContext(
        rank=rank,
        node=node,
        num_ranks=num_ranks,
        num_nodes=num_nodes,
        signal_tensor=signal_tensor,
        ll_buffers=ll_buffers,
        grid_barrier=grid_barrier,
        max_buffer_size=max_buffer_size,
        signal_target=15,
    )

    return ctx


def fast_allgather_pull(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_target += 1
    return _forward_pull_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.signal_tensor,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )


def fast_allgather_push_2d(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_target += 1
    _forward_push_2d_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.signal_tensor,
        ctx.num_nodes,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )
    return symm_buffer


def fast_allgather_push_3d(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    ctx.signal_target += 1
    _forward_push_3d_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.ll_buffers[ctx.signal_target % 2],
        ctx.signal_tensor,
        ctx.num_nodes,
        2,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        INTER_NODE_WITH_LL=False,
        num_warps=32,
    )
    return symm_buffer


def fast_allgather_push_2d_ll(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    ctx.signal_target += 1
    ll_buffer = ctx.ll_buffers[ctx.signal_target % 2]
    _forward_push_2d_ll_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ctx.signal_tensor,
        ll_buffer,
        ctx.num_nodes,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )

    return symm_buffer


def fast_allgather_push_2d_ll_multimem(
    ctx: FastAllGatherContext, symm_buffer: torch.Tensor
):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    ctx.signal_target += 1
    ll_buffer = ctx.ll_buffers[ctx.signal_target % 2]
    _forward_push_2d_ll_multimem_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        ll_buffer,
        ctx.num_nodes,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )

    return symm_buffer


def fast_allgather_push_numa_2d(ctx: FastAllGatherContext, symm_buffer: torch.Tensor):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    signal = ctx.signal[ctx.signal_target % 2]
    _forward_push_numa_2d_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        signal,
        2,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )
    ctx.signal_target += 1
    return symm_buffer


def fast_allgather_push_numa_2d_ll(
    ctx: FastAllGatherContext, symm_buffer: torch.Tensor
):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    assert ctx.num_nodes == 1
    signal = ctx.signal[ctx.signal_target % 2]
    _forward_push_numa_2d_ll_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        signal,
        ctx.ll_buffers[ctx.signal_target % 2],
        2,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )
    ctx.signal_target += 1
    return symm_buffer


def fast_allgather_push_numa_2d_ll_multinode(
    ctx: FastAllGatherContext, symm_buffer: torch.Tensor
):
    assert symm_buffer.nbytes * 2 < ctx.max_buffer_size
    signal = ctx.signal[ctx.signal_target % 2]
    _forward_push_numa_2d_ll_multinode_kernel[(ctx.num_ranks,)](
        symm_buffer,
        symm_buffer.nbytes // ctx.num_ranks,
        signal,
        ctx.ll_buffers[ctx.signal_target % 2],
        ctx.num_nodes,
        2,
        ctx.num_ranks,
        ctx.rank,
        ctx.signal_target,
        num_warps=32,
    )
    ctx.signal_target += 1
    return symm_buffer


FAST_ALLGATHER_FUNC_DISPATCH = {
    "pull": fast_allgather_pull,
    "push2d": fast_allgather_push_2d,
    "push2d_ll": fast_allgather_push_2d_ll,
    "push2d_ll_multimem": fast_allgather_push_2d_ll_multimem,
    "push_numa_2d": fast_allgather_push_numa_2d,
    "push_numa_2d_ll": fast_allgather_push_numa_2d_ll,
    "push_numa_2d_ll_multinode": fast_allgather_push_numa_2d_ll_multinode,
}


def fast_allgather(
    symm_buffer: torch.Tensor,
    ctx: FastAllGatherContext = None,
    rank=None,
    node=None,
    num_ranks=None,
    num_nodes=None,
    mode="pull",
):
    assert mode in FAST_ALLGATHER_FUNC_DISPATCH
    if ctx is None:
        assert rank is not None and node is not None
        assert num_ranks is not None and num_nodes is not None
        ctx = create_fast_allgather_context(
            rank,
            node,
            num_ranks,
            num_nodes,
        )
    return FAST_ALLGATHER_FUNC_DISPATCH[mode](ctx, symm_buffer)
