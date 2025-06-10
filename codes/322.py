import math
import typing as tp
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy
import triton
import triton.language as tl

F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


def safe_autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
) -> tp.Callable[[F], F]:

    try:
        from triton.runtime.autotuner import Autotuner

        def decorator(fn):
            return Autotuner(
                fn,
                fn.arg_names,
                configs,
                key,
                reset_to_zero,
                restore_value,
                pre_hook=pre_hook,
                post_hook=post_hook,
                prune_configs_by=prune_configs_by,
                warmup=warmup,
                rep=rep,
                use_cuda_graph=use_cuda_graph,
            )

        return decorator
    except Exception as err:
        print(f"Couldn't autotune given function due to {err}")

        def decorator(fn):
            return fn

        return decorator


def dtype_index(x: jnp.array) -> int:
    if x.dtype == jnp.float16:
        return 1
    if x.dtype == jnp.bfloat16:
        return 2
    if x.dtype == jnp.float32:
        return 3
    raise ValueError(x.dtype)


def get_sharding(arr: chex.Array):

    return getattr(arr, "sharding", None)


def get_strides(shape: tuple[int, ...]) -> tuple[int, ...]:

    if hasattr(shape, "shape"):
        shape = shape.shape
    size = numpy.prod(shape)
    strides = []
    for s in shape:
        size = int(size // s)
        strides.append(size)
    return tuple(strides)


@triton.jit
def padded_load(
    ptrs,
    offs_a,
    offs_b,
    PA0: tl.constexpr,
    PA1: tl.constexpr,
    LA0: tl.constexpr,
    LA1: tl.constexpr,
):
    if PA0:
        if PA1:
            x = tl.load(
                ptrs,
                mask=(offs_a[:, None] < LA0) & (offs_b[None, :] < LA1),
                other=0.0,
            )
        else:
            x = tl.load(
                ptrs,
                mask=offs_a[:, None] < LA0,
                other=0.0,
            )
    else:
        if PA1:
            x = tl.load(
                ptrs,
                mask=offs_b[None, :] < LA1,
                other=0.0,
            )
        else:
            x = tl.load(ptrs)
    return x


def calc_bias_strides(
    bias: tp.Optional[jnp.ndarray],
    batch: int,
    nheads_q: int,
    QSeq: int,
    KSeq: int,
) -> tp.Tuple[int, ...]:
    if bias is not None:
        if not hasattr(bias, "strides"):
            strides = tuple(map(lambda x: x * bias.itemsize, get_strides(bias)))
        else:
            strides = bias.strides
        if bias.shape[2] != QSeq or bias.shape[3] != KSeq:
            raise ValueError(
                f"Bias tensor has incompatible sequence dimensions. "
                f"Expected shape [..., {QSeq}, {KSeq}], but got [..., {bias.shape[2]}, {bias.shape[3]}]. "
                f"Full bias shape: {bias.shape}"
            )
        if bias.shape[0] == 1:
            stride_bz = 0
        elif bias.shape[0] == batch:
            stride_bz = strides[0] // bias.itemsize
        else:
            raise ValueError(
                f"Batch dimension mismatch in bias tensor. "
                f"Expected either 1 (for broadcasting) or {batch} (batch size), "
                f"but got {bias.shape[0]}. Consider reshaping your bias tensor."
            )
        if bias.shape[1] == 1:
            stride_bh = 0
        elif bias.shape[1] == nheads_q:
            stride_bh = strides[1] // bias.itemsize
        else:
            raise ValueError(
                f"Head dimension mismatch in bias tensor. "
                f"Expected either 1 (for broadcasting) or {nheads_q} (number of heads), "
                f"but got {bias.shape[1]}. Check that your bias tensor matches the model configuration."
            )

        stride_bm = strides[2] // bias.itemsize
    else:
        stride_bz, stride_bh, stride_bm = 0, 0, 0
    return stride_bz, stride_bh, stride_bm


@partial(jax.jit, static_argnames=["max_tokens"])
def attention_pack_with_static_shape(
    x: jnp.ndarray,
    attention_mask: jnp.ndarray,
    max_tokens: int = None,
) -> jnp.ndarray:

    batch_size, seqlen = attention_mask.shape
    num_heads, head_dim = x.shape[2], x.shape[3]

    if max_tokens is None:
        max_tokens = batch_size * seqlen

    seqlens = jnp.sum(attention_mask, axis=1).astype(jnp.int32)
    offsets = jnp.zeros((batch_size,), dtype=jnp.int32)
    offsets = offsets.at[1:].set(jnp.cumsum(seqlens[:-1]))
    packed = jnp.zeros((1, max_tokens, num_heads, head_dim), dtype=x.dtype)
    batch_idx, pos_idx = jnp.meshgrid(
        jnp.arange(batch_size), jnp.arange(seqlen), indexing="ij"
    )

    batch_idx_flat = batch_idx.reshape(-1)
    pos_idx_flat = pos_idx.reshape(-1)

    valid_mask = pos_idx < seqlens[:, None]
    target_idx = jnp.where(
        valid_mask,
        offsets[:, None] + pos_idx,
        jnp.zeros_like(pos_idx),
    )
    target_idx_flat = target_idx.reshape(-1)
    valid_mask_flat = valid_mask.reshape(-1)

    def process_token(i, packed_acc):
        b = batch_idx_flat[i]
        p = pos_idx_flat[i]
        t = target_idx_flat[i]
        valid = valid_mask_flat[i]
        packed_acc = jnp.where(valid, packed_acc.at[0, t].set(x[b, p]), packed_acc)

        return packed_acc

    packed = jax.lax.fori_loop(0, batch_size * seqlen, process_token, packed)
    return packed


@partial(jax.jit, static_argnames=["seqlen", "batch_size"])
def attention_unpack_with_static_shape(
    x: jnp.ndarray,
    cum_seqlens: jnp.ndarray,
    batch_size: int,
    seqlen: int,
) -> jnp.ndarray:

    num_heads, head_dim = x.shape[2], x.shape[3]

    unpacked = jnp.zeros((batch_size, seqlen, num_heads, head_dim), dtype=x.dtype)

    def process_batch(b, unpacked_acc):
        start_idx = cum_seqlens[b]
        end_idx = cum_seqlens[b + 1]
        seq_len = end_idx - start_idx

        def process_position(p, acc):

            valid = p < seq_len
            src_idx = start_idx + p

            acc = jnp.where(valid, acc.at[b, p].set(x[0, src_idx]), acc)

            return acc

        unpacked_acc = jax.lax.fori_loop(0, seqlen, process_position, unpacked_acc)

        return unpacked_acc

    unpacked = jax.lax.fori_loop(0, batch_size, process_batch, unpacked)

    return unpacked


def basic_attention_refrence(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_bias: tp.Optional[jnp.ndarray] = None,
    query_padding_mask: tp.Optional[jnp.ndarray] = None,
    key_padding_mask: tp.Optional[jnp.ndarray] = None,
    dropout_prob: float = 0.0,
    dropout_key: tp.Optional[jax.random.PRNGKey] = None,
    window_size: tp.Tuple[int, int] = (-1, -1),
    causal: bool = False,
    softcap: float = 0.0,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    QSeq, KSeq = q.shape[1], k.shape[1]
    repeats = q.shape[2] // k.shape[2]
    if repeats > 1:
        k = jnp.repeat(k, repeats=repeats, axis=2)
        v = jnp.repeat(v, repeats=repeats, axis=2)
    d = q.shape[-1]
    q_scaled = q / math.sqrt(d)
    scores = jnp.einsum("bthd,bshd->bhts", q_scaled, k)
    if softcap > 0:
        scores = scores / softcap
        scores = jnp.tanh(scores)
        scores = scores * softcap
    if key_padding_mask is not None:
        key_mask = (~key_padding_mask).reshape(key_padding_mask.shape[0], 1, 1, KSeq)
        scores = jnp.where(key_mask, jnp.finfo(scores.dtype).min, scores)
    if window_size[0] >= 0 or window_size[1] >= 0:
        row_idx = jnp.arange(QSeq).reshape(-1, 1)
        col_idx = jnp.arange(KSeq)
        if key_padding_mask is None:
            sk = KSeq
        else:
            sk = jnp.sum(key_padding_mask, axis=-1).reshape(-1, 1, 1, 1)
        if query_padding_mask is None:
            sq = QSeq
        else:
            sq = jnp.sum(query_padding_mask, axis=-1).reshape(-1, 1, 1, 1)
        if window_size[0] < 0:
            local_mask = col_idx > row_idx + sk - sq + window_size[1]
        else:
            if key_padding_mask is None:
                sk_full = jnp.full_like(col_idx, KSeq)
            else:
                sk_full = sk
            local_mask = jnp.logical_or(
                col_idx > jnp.minimum(row_idx + sk - sq + window_size[1], sk_full),
                col_idx < row_idx + sk - sq - window_size[0],
            )
        scores = jnp.where(local_mask, jnp.finfo(scores.dtype).min, scores)
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    if window_size[0] >= 0 or window_size[1] >= 0:
        all_masked = jnp.all(local_mask, axis=-1, keepdims=True)
        attention = jnp.where(all_masked, 0.0, attention)
    if query_padding_mask is not None:
        query_mask = (~query_padding_mask).reshape(
            query_padding_mask.shape[0], 1, QSeq, 1
        )
        attention = jnp.where(query_mask, 0.0, attention)
    dropout_scaling = 1.0 / (1 - dropout_prob)
    if dropout_prob > 0 and dropout_key is not None:
        dropout_mask = jax.random.bernoulli(
            dropout_key, p=1 - dropout_prob, shape=attention.shape
        )
        attention_drop = attention * dropout_mask * dropout_scaling
    else:
        attention_drop = attention
    output = jnp.einsum("bhts,bshd->bthd", attention_drop, v)
    if query_padding_mask is not None:
        query_mask_expanded = (~query_padding_mask).reshape(
            query_padding_mask.shape[0],
            QSeq,
            1,
            1,
        )
        output = jnp.where(query_mask_expanded, 0.0, output)
    return output.astype(dtype_og)
