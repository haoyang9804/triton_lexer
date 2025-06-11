import math

import torch
import triton
import triton.language as tl


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu_new(x):
    pi = math.pi
    a = tl.math.sqrt(2.0 / pi)
    b = x + 0.044715 * x * x * x
    return 0.5 * x * (1.0 + tanh(a * b))


@triton.jit
def dropout(x, p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random > p, x / (1 - p), 0.0)


@triton.jit
def fused_embeddings_kernel(
    x_ptr,
    wte_ptr,
    wpe_ptr,
    z_ptr,
    B,
    L,
    V,
    P,
    H,
    dropout_prob=0.0,
    seed=1337,
    BLOCK_SIZE: tl.constexpr = 512,
):

    pid = tl.program_id(0)
    wte_ptr += tl.load(x_ptr + pid) * H
    wpe_ptr += (pid % L) * H
    z_ptr += pid * H

    for k in range(0, H, BLOCK_SIZE):
        offset = k + tl.arange(0, BLOCK_SIZE)
        mask = offset < H

        z = tl.load(wte_ptr + offset, mask=mask, other=0.0)
        z += tl.load(wpe_ptr + offset, mask=mask, other=0.0)
        z = dropout(z, dropout_prob, seed, offset)

        tl.store(z_ptr + offset, z, mask=mask)


@torch.no_grad()
def fused_embeddings(x, wte, wpe, dropout_prob=0.0):

    assert wte.shape[1] == wpe.shape[1]
    assert x.is_contiguous()
    assert wte.is_contiguous()
    assert wpe.is_contiguous()
    B, L = x.shape
    V, H = wte.shape
    P = wpe.shape[0]
    z = torch.empty((B * L, H), device=x.device, dtype=wte.dtype)
    grid = (z.shape[0],)
    fused_embeddings_kernel[grid](
        x.view(-1),
        wte,
        wpe,
        z,
        B,
        L,
        V,
        P,
        H,
        dropout_prob=dropout_prob,
    )
    return z.view((B, L, H))


@triton.jit
def fused_layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, z_ptr, H, eps=1e-5, BLOCK_SIZE: tl.constexpr = 512
):

    row_id = tl.program_id(0)
    x_ptr += row_id * H
    z_ptr += row_id * H

    x_mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, H, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offset, mask=(offset < H), other=0.0)
        x_mean += x.to(tl.float32)
    x_mean = tl.sum(x_mean) / H

    x_var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, H, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offset, mask=(offset < H), other=x_mean)
        x = x.to(tl.float32)
        x_var += (x - x_mean) * (x - x_mean)
    x_var = tl.sum(x_var) / H
    rstd = 1 / tl.sqrt(x_var + eps)

    for i in range(0, H, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < H

        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        w = tl.load(w_ptr + offset, mask=mask, other=0.0)
        b = tl.load(b_ptr + offset, mask=mask, other=0.0)

        z = (x - x_mean) * rstd
        z = z * w + b

        tl.store(z_ptr + offset, z, mask=mask)


@torch.no_grad()
def fused_layer_norm(x, weight, bias):

    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()
    assert weight.shape == bias.shape
    assert x.shape[-1] == weight.shape[0]
    out_shape = x.shape
    x = x.view((-1, x.shape[-1]))
    B, H = x.shape
    x = x.view((B, H))
    z = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    fused_layer_norm_kernel[(B,)](x, weight, bias, z, H)
    return z.view(out_shape)


@triton.jit
def fused_ffn_kernel(
    x_ptr,
    w_ptr,
    z_ptr,
    M,
    N,
    K,
    b_ptr=None,
    r_ptr=None,
    apply_gelu=False,
    dropout_prob=0.0,
    seed=1337,
    BLOCK_SIZE_M: tl.constexpr = 128,
    BLOCK_SIZE_N: tl.constexpr = 128,
    BLOCK_SIZE_K: tl.constexpr = 64,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]

    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        x = tl.load(x_ptr + offs_m * K + x_k, mask=(offs_m < M) & (x_k < K), other=0.0)

        x = x.to(tl.float16)

        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        w = tl.load(w_ptr + w_k * N + offs_n, mask=(w_k < K) & (offs_n < N), other=0.0)
        w = w.to(tl.float16)

        z = tl.dot(x, w, acc=z)

    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0)
        z += b.to(tl.float32)

    z_offset = offs_m * N + offs_n
    z_mask = (offs_m < M) & (offs_n < N)

    if apply_gelu:
        z = gelu_new(z)
    if dropout_prob > 0.0:
        z = dropout(z, dropout_prob, seed, z_offset)

    if r_ptr is not None:
        r = tl.load(r_ptr + z_offset, mask=z_mask)
        z += r.to(tl.float32)

    tl.store(z_ptr + z_offset, z, mask=z_mask)


@torch.no_grad()
def fused_ffn(
    x,
    weight,
    bias=None,
    residual=None,
    add_gelu=False,
    dropout_prob=0.0,
):

    out_shape_0 = x.shape[:-1]
    x = x.view((-1, x.shape[-1]))

    M, K = x.shape
    N = weight.shape[1]

    x = x.view((M, K))
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)

    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert x.shape[1] == weight.shape[0]
    if bias is not None:
        assert bias.is_contiguous()
        assert weight.shape[1] == bias.shape[0]
    if residual is not None:
        residual = residual.view(z.shape)
        assert residual.is_contiguous()

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    fused_ffn_kernel[grid](
        x,
        weight,
        z,
        M,
        N,
        K,
        apply_gelu=add_gelu,
        dropout_prob=dropout_prob,
        b_ptr=bias,
        r_ptr=residual,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=8,
    )
    return z.view((*out_shape_0, N))


@torch.no_grad()
def matmul_and_split_qkv(x, weight, bias, num_heads):

    x = fused_ffn(x, weight, bias=bias)

    batch_size, seqlen, hidden_size = x.shape
    assert hidden_size % 3 == 0, hidden_size
    hidden_size = hidden_size // 3
    q, k, v = x.split(hidden_size, dim=2)
    assert hidden_size % num_heads == 0, (hidden_size, num_heads)
    head_size = hidden_size // num_heads

    q, k, v = map(
        lambda x: x.view(batch_size, seqlen, num_heads, head_size)
        .transpose(1, 2)
        .contiguous(),
        (q, k, v),
    )

    return q, k, v


@triton.jit
def flash_attention_v1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    z_ptr,
    BN,
    Lq,
    Lk,
    scale,
    H: tl.constexpr,
    dropout_prob=0.0,
    seed=1337,
    BLOCK_SIZE_L: tl.constexpr = 64,
):

    q_ptr += tl.program_id(0) * (Lq * H)
    z_ptr += tl.program_id(0) * (Lq * H)
    k_ptr += tl.program_id(0) * (Lk * H)
    v_ptr += tl.program_id(0) * (Lk * H)

    offs_lq = tl.program_id(1) * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_h = tl.arange(0, H)

    q_mask = offs_lq[:, None] < Lq
    q_offs = offs_lq[:, None] * H + offs_h[None, :]

    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)

    q = q.to(tl.float16)

    z = tl.zeros((BLOCK_SIZE_L, H), dtype=tl.float32)
    max_value = tl.zeros((BLOCK_SIZE_L, 1), dtype=tl.float32) + float("-inf")
    denominator = tl.zeros((BLOCK_SIZE_L, 1), dtype=tl.float32)
    for i in range(0, Lk, BLOCK_SIZE_L):
        offs_lk = i + tl.arange(0, BLOCK_SIZE_L)
        kv_mask = offs_lk[:, None] < Lk
        kv_offs = offs_lk[:, None] * H + offs_h[None, :]

        k = tl.load(k_ptr + kv_offs, mask=kv_mask, other=0.0)

        k = k.to(q.dtype)
        qk = tl.dot(q, k.trans(1, 0)) * scale

        qk = tl.where(offs_lq[:, None] >= offs_lk[None, :], qk, float("-inf"))

        block_max_value = tl.max(qk, axis=1, keep_dims=True)

        new_max_value = tl.where(
            block_max_value > max_value, block_max_value, max_value
        )

        qk = tl.exp(qk - new_max_value)

        multiplier = tl.exp(max_value - new_max_value)
        denominator *= multiplier
        z *= multiplier

        denominator += tl.sum(qk, axis=1, keep_dims=True)
        max_value = new_max_value

        if dropout_prob > 0.0:
            qk_offs = offs_lq[:, None] * Lk + offs_lk[None, :]
            qk = dropout(qk, dropout_prob, seed, qk_offs)

        v = tl.load(v_ptr + kv_offs, mask=kv_mask, other=0.0)

        v = v.to(q.dtype)
        qk = qk.to(q.dtype)

        z = tl.dot(qk, v, acc=z)

    z /= denominator
    z = z.to(z_ptr.dtype.element_ty)

    tl.store(z_ptr + q_offs, z, mask=q_mask)


@torch.no_grad()
def flash_attention_v1(q, k, v, dropout_prob=0.0):

    assert q.shape[:2] == k.shape[:2]
    assert q.shape[-1] == k.shape[-1]
    assert k.shape == v.shape

    B, N, Lq, H = q.shape
    Lk = k.shape[2]

    assert H in {16, 32, 64, 128, 256}

    q = q.view(B * N, Lq, H)
    k = k.view(B * N, Lk, H)
    v = v.view(B * N, Lk, H)

    z = torch.empty_like(q)

    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert z.is_contiguous()

    scale = 1 / math.sqrt(H)

    BLOCK_SIZE_L = 64
    grid = (B * N, triton.cdiv(Lq, BLOCK_SIZE_L), 1)
    flash_attention_v1_kernel[grid](
        q,
        k,
        v,
        z,
        B * N,
        Lq,
        Lk,
        scale,
        H,
        dropout_prob=dropout_prob,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    return z.view(B, N, Lq, H)
