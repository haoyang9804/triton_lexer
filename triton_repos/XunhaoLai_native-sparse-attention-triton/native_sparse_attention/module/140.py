import torch
import triton
import triton.language as tl
from typing import Union
from native_sparse_attention.ops.triton.utils import get_compressed_seqlens


class KVCache:
    def __init__(
        self,
        max_batch_size: int,
        max_length: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ):
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.kv_cache = torch.zeros(
            2,
            self.max_batch_size,
            self.max_length,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_len = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

    def reset(self):
        self.kv_cache.zero_()
        self.kv_len.zero_()

    def update_kv(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        if step == 0:
            self._update_kv_prefill(
                cu_seqlens,
                step,
                key,
                value,
            )
        else:
            self._update_kv_decode(
                cu_seqlens,
                step,
                key,
                value,
            )

    def _update_kv_prefill(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        assert step == 0
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = seqlens.shape[0]

        total_len, num_heads, head_dim = key.shape
        assert key.shape == value.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim
        assert total_len == cu_seqlens[-1].item()

        seq_start, seq_end = cu_seqlens[:-1], cu_seqlens[1:]
        _fill_kv_cache(
            self.kv_cache,
            key,
            value,
            seq_start,
            seq_end,
        )
        self.kv_len[:batch_size] = seqlens

    def _update_kv_decode(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        assert step > 0
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        batch_size, num_heads, head_dim = key.shape
        assert batch_size == seqlens.shape[0]
        assert key.shape == value.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim

        brange = torch.arange(batch_size, dtype=torch.int32, device=key.device)
        self.kv_cache[0, :batch_size][brange, self.kv_len[:batch_size]] = key
        self.kv_cache[1, :batch_size][brange, self.kv_len[:batch_size]] = value
        self.kv_len[:batch_size] += 1


class NSACache:

    def __init__(
        self,
        max_batch_size: int,
        max_length: int,
        num_kv_heads: int,
        head_dim: int,
        kernel_size: int,
        kernel_stride: int,
        window_size: int,
        dtype: torch.dtype,
        device: Union[str, torch.device] = "cuda",
    ):
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.window_size = window_size
        self.dtype = dtype
        self.device = device

        self.sparse_kv_cache = torch.zeros(
            2,
            self.max_batch_size,
            self.max_length,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.sparse_kv_len = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

        self.max_comp_length = (
            self.max_length - self.kernel_size
        ) // self.kernel_stride + 1
        self.compress_kv_cache = torch.zeros(
            2,
            self.max_batch_size,
            self.max_comp_length,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.compress_kv_len = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )
        self.before_compress_kv_cache = torch.zeros(
            2,
            self.max_batch_size,
            self.kernel_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.before_compress_kv_len = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

        self.sliding_kv_cache = torch.zeros(
            2,
            self.max_batch_size,
            self.window_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.sliding_kv_len = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

    def reset(self):
        self.compress_kv_cache.zero_()
        self.compress_kv_len.zero_()
        self.before_compress_kv_cache.zero_()
        self.before_compress_kv_len.zero_()
        self.sparse_kv_cache.zero_()
        self.sparse_kv_len.zero_()
        self.sliding_kv_cache.zero_()
        self.sliding_kv_len.zero_()

    def prepare_compress(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        if step == 0:
            self._prepare_compress_prefill(cu_seqlens, step, key, value)
        else:
            self._prepare_compress_decode(cu_seqlens, step, key, value)

    def _prepare_compress_prefill(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        assert step == 0

        batch_size = cu_seqlens.shape[0] - 1
        total_len, num_heads, head_dim = key.shape
        assert key.shape == value.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim
        comp_seqlens, comp_cu_seqlens = get_compressed_seqlens(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )
        assert total_len == cu_seqlens[-1].item()

        seq_start = cu_seqlens[:-1] + comp_seqlens * self.kernel_stride
        seq_end = cu_seqlens[1:]
        _fill_kv_cache(
            self.before_compress_kv_cache,
            key,
            value,
            seq_start,
            seq_end,
        )
        self.before_compress_kv_len[:batch_size] = seq_end - seq_start

    def _prepare_compress_decode(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        assert step > 0

        batch_size, num_heads, head_dim = key.shape
        assert key.shape == value.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim
        assert batch_size == cu_seqlens.shape[0] - 1

        idx = torch.where(self.before_compress_kv_len == self.kernel_size)[0].squeeze()
        self.before_compress_kv_cache[
            :, idx, : self.kernel_size - self.kernel_stride, :, :
        ] = self.before_compress_kv_cache[:, idx, self.kernel_stride :, :, :]
        self.before_compress_kv_len[idx] -= self.kernel_stride

        brange = torch.arange(batch_size, dtype=torch.int32, device=key.device)
        self.before_compress_kv_cache[0, :batch_size][
            brange, self.before_compress_kv_len[:batch_size]
        ] = key
        self.before_compress_kv_cache[1, :batch_size][
            brange, self.before_compress_kv_len[:batch_size]
        ] = value

        self.before_compress_kv_len[:batch_size] += 1

    def update_kv(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        compress_key: torch.Tensor,
        compress_value: torch.Tensor,
        sparse_key: torch.Tensor,
        sparse_value: torch.Tensor,
        sliding_key: torch.Tensor,
        sliding_value: torch.Tensor,
    ):
        if step == 0:
            self._update_kv_prefill(
                cu_seqlens,
                step,
                compress_key,
                compress_value,
                sparse_key,
                sparse_value,
                sliding_key,
                sliding_value,
            )
        else:
            self._update_kv_decode(
                cu_seqlens,
                step,
                compress_key,
                compress_value,
                sparse_key,
                sparse_value,
                sliding_key,
                sliding_value,
            )

    def _update_kv_prefill(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        compress_key: torch.Tensor,
        compress_value: torch.Tensor,
        sparse_key: torch.Tensor,
        sparse_value: torch.Tensor,
        sliding_key: torch.Tensor,
        sliding_value: torch.Tensor,
    ):
        assert step == 0
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_size = seqlens.shape[0]

        total_len, num_heads, head_dim = sparse_key.shape
        assert sparse_key.shape == sparse_value.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim
        assert total_len == cu_seqlens[-1].item()

        total_comp_len, num_heads, head_dim = compress_key.shape
        assert compress_key.shape == compress_value.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim
        comp_seqlens, comp_cu_seqlens = get_compressed_seqlens(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )
        assert total_comp_len == comp_cu_seqlens[-1].item()

        total_len, num_heads, head_dim = sliding_key.shape
        assert sliding_key.shape == sliding_value.shape

        seq_start, seq_end = comp_cu_seqlens[:-1], comp_cu_seqlens[1:]
        _fill_kv_cache(
            self.compress_kv_cache,
            compress_key,
            compress_value,
            seq_start,
            seq_end,
        )
        self.compress_kv_len[:batch_size] = comp_seqlens

        seq_start, seq_end = cu_seqlens[:-1], cu_seqlens[1:]
        _fill_kv_cache(
            self.sparse_kv_cache,
            sparse_key,
            sparse_value,
            seq_start,
            seq_end,
        )
        self.sparse_kv_len[:batch_size] = seqlens

        seq_start = torch.maximum(cu_seqlens[1:] - self.window_size, cu_seqlens[:-1])
        seq_end = cu_seqlens[1:]
        _fill_kv_cache(
            self.sliding_kv_cache,
            sliding_key,
            sliding_value,
            seq_start,
            seq_end,
        )
        self.sliding_kv_len[:batch_size] = seq_end - seq_start

    def _update_kv_decode(
        self,
        cu_seqlens: torch.Tensor,
        step: int,
        compress_key: torch.Tensor,
        compress_value: torch.Tensor,
        sparse_key: torch.Tensor,
        sparse_value: torch.Tensor,
        sliding_key: torch.Tensor,
        sliding_value: torch.Tensor,
    ):
        assert step > 0
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        batch_size, num_heads, head_dim = sparse_key.shape
        assert batch_size == seqlens.shape[0]
        assert sparse_key.shape == sparse_value.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim

        batch_size, num_heads, head_dim = compress_key.shape
        assert batch_size == seqlens.shape[0]
        assert compress_key.shape == compress_value.shape
        assert num_heads == self.num_kv_heads and head_dim == self.head_dim

        total_len, num_heads, head_dim = sliding_key.shape
        assert sliding_key.shape == sliding_value.shape

        idx = torch.where(self.before_compress_kv_len == self.kernel_size)[0].squeeze()
        self.compress_kv_cache[0][idx, self.compress_kv_len[idx]] = compress_key[idx]
        self.compress_kv_cache[1][idx, self.compress_kv_len[idx]] = compress_value[idx]
        self.compress_kv_len[idx] += 1

        brange = torch.arange(batch_size, dtype=torch.int32, device=sparse_key.device)
        self.sparse_kv_cache[0, :batch_size][
            brange, self.sparse_kv_len[:batch_size]
        ] = sparse_key
        self.sparse_kv_cache[1, :batch_size][
            brange, self.sparse_kv_len[:batch_size]
        ] = sparse_value
        self.sparse_kv_len[:batch_size] += 1

        self.sliding_kv_cache[0, :batch_size][
            brange, self.sliding_kv_len[:batch_size] % self.window_size
        ] = sliding_key
        self.sliding_kv_cache[1, :batch_size][
            brange, self.sliding_kv_len[:batch_size] % self.window_size
        ] = sliding_value
        self.sliding_kv_len[:batch_size] += 1


@triton.jit
def _fill_kv_cache_kernel(
    cache_ptr,
    k_ptr,
    v_ptr,
    seq_start,
    seq_end,
    head_dim,
    stride_c2,
    stride_cb,
    stride_cn,
    stride_ch,
    stride_cd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):

    pid_2b = tl.program_id(0)
    pid_2 = pid_2b % 2
    pid_b = pid_2b // 2
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    kv_start = tl.load(seq_start + pid_b)
    kv_end = tl.load(seq_end + pid_b)
    kv_len = kv_end - kv_start
    if pid_n * BLOCK_SIZE_N >= kv_len:
        return

    if pid_2 == 0:
        kv_ptrs = tl.make_block_ptr(
            base=k_ptr + kv_start * stride_kn + pid_h * stride_kh,
            shape=(kv_len, head_dim),
            strides=(stride_kn, stride_kd),
            offsets=(pid_n * BLOCK_SIZE_N, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(1, 0),
        )
        kv = tl.load(kv_ptrs, boundary_check=(0, 1))
    else:
        kv_ptrs = tl.make_block_ptr(
            base=v_ptr + kv_start * stride_vn + pid_h * stride_vh,
            shape=(kv_len, head_dim),
            strides=(stride_vn, stride_vd),
            offsets=(pid_n * BLOCK_SIZE_N, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(1, 0),
        )
        kv = tl.load(kv_ptrs, boundary_check=(0, 1))

    cache_ptrs = tl.make_block_ptr(
        base=cache_ptr + pid_2 * stride_c2 + pid_b * stride_cb + pid_h * stride_ch,
        shape=(kv_len, head_dim),
        strides=(stride_cn, stride_cd),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(cache_ptrs, kv.to(cache_ptr.dtype.element_ty), boundary_check=(0, 1))


def _fill_kv_cache(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_start: torch.Tensor,
    seq_end: torch.Tensor,
):
    total_len, num_heads, head_dim = key.shape
    batch_size = seq_start.shape[0]
    max_kv_len = (seq_end - seq_start).max().item()

    if max_kv_len == 0:
        return
    BLOCK_SIZE_N = min(1024, triton.next_power_of_2(max_kv_len))
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    grid = (2 * batch_size, num_heads, triton.cdiv(max_kv_len, BLOCK_SIZE_N))
    _fill_kv_cache_kernel[grid](
        kv_cache,
        key,
        value,
        seq_start,
        seq_end,
        head_dim,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
