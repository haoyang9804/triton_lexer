import torch
import triton
import triton.language as tl


@triton.jit
def parallel_rebased_fwd_kernel(
    q,
    k,
    v,
    o,
    z,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    scale,
    B,
    H,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):

    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(
        q + i_bh * s_qk_h,
        (T, K),
        (s_qk_t, s_qk_d),
        (i_c * BTL, i_k * BK),
        (BTL, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1)
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    for _ in range(0, i_c * BTL, BTS):

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)

        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    tl.debug_barrier()
    o_q = tl.arange(0, BTL)

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(
        k + i_bh * s_qk_h,
        (K, T),
        (s_qk_d, s_qk_t),
        (i_k * BK, i_c * BTL),
        (BK, BTS),
        (0, 1),
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_vo_h,
        (T, V),
        (s_vo_t, s_vo_d),
        (i_c * BTL, i_v * BV),
        (BTS, BV),
        (1, 0),
    )

    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))

        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)

        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(
        o + (i_bh + B * H * i_k) * s_vo_h,
        (T, V),
        (s_vo_t, s_vo_d),
        (i_c * BTL, i_v * BV),
        (BTL, BV),
        (1, 0),
    )
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(
        p_z, b_z.to(p_z.dtype.element_ty), mask=((i_c * BTL + tl.arange(0, BTL)) < T)
    )


@triton.jit
def _parallel_rebased_bwd_dq(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dq,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    p_do = tl.make_block_ptr(
        do + i_bh * s_v_h,
        (T, V),
        (s_v_t, s_v_d),
        (i_c * BTL, i_v * BV),
        (BTL, BV),
        (1, 0),
    )
    p_q = tl.make_block_ptr(
        q + (i_bh) * s_k_h,
        (T, K),
        (s_k_t, s_k_d),
        (i_c * BTL, i_k * BK),
        (BTL, BK),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(
        k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (0, i_k * BK), (BTS, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, 0), (BV, BTS), (0, 1)
    )
    p_dz = dz + i_bh * T + i_c * BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c * BTL + tl.arange(0, BTL)) < T)

    for _ in range(0, i_c * BTL, BTS):

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)

        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(
        k + i_bh * s_k_h,
        (T, K),
        (s_k_t, s_k_d),
        (i_c * BTL, i_k * BK),
        (BTS, BK),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_v_h,
        (V, T),
        (s_v_d, s_v_t),
        (i_v * BV, i_c * BTL),
        (BV, BTS),
        (0, 1),
    )

    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))

        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)

        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(
        dq + (i_bh + B * H * i_v) * s_k_h,
        (T, K),
        (s_k_t, s_k_d),
        (i_c * BTL, i_k * BK),
        (BTL, BK),
        (1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit
def _parallel_rebased_bwd_dkv(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dk,
    dv,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):

    p_k = tl.make_block_ptr(
        k + i_bh * s_k_h,
        (T, K),
        (s_k_t, s_k_d),
        (i_c * BTL, i_k * BK),
        (BTL, BK),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_v_h,
        (T, V),
        (s_v_t, s_v_d),
        (i_c * BTL, i_v * BV),
        (BTL, BV),
        (1, 0),
    )
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros(
        [BTL, BV], dtype=tl.float32
    )

    for i in range((tl.cdiv(T, BTS) * BTS) - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(
            q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i), (BK, BTS), (0, 1)
        )
        p_do = tl.make_block_ptr(
            do + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i), (BV, BTS), (0, 1)
        )
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
        b_s = tl.dot(b_k.to(b_q.dtype), b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)

    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        p_q = tl.make_block_ptr(
            q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i), (BK, BTS), (0, 1)
        )
        p_do = tl.make_block_ptr(
            do + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i), (BV, BTS), (0, 1)
        )
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)

        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)

        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale

        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        o_q += BTS

    p_dk = tl.make_block_ptr(
        dk + (i_bh + B * H * i_v) * s_k_h,
        (T, K),
        (s_k_t, s_k_d),
        (i_c * BTL, i_k * BK),
        (BTL, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (i_bh + B * H * i_k) * s_v_h,
        (T, V),
        (s_v_t, s_v_d),
        (i_c * BTL, i_v * BV),
        (BTL, BV),
        (1, 0),
    )
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit
def parallel_rebased_bwd_kernel(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(
        i_bh,
        i_c,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dz,
        dq,
        s_qk_h,
        s_qk_t,
        s_qk_d,
        s_vo_h,
        s_vo_t,
        s_vo_d,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
    )
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh,
        i_c,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dz,
        dk,
        dv,
        s_qk_h,
        s_qk_t,
        s_qk_d,
        s_vo_h,
        s_vo_t,
        s_vo_d,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
    )


class ParallelBasedFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        BTL, BTS = 128, 32
        assert BTL % BTS == 0
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        BK, BV = max(BK, 16), max(BV, 16)
        B, H, T, K, V = *k.shape, v.shape[-1]
        num_stages = 2
        num_warps = 4
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        grid = (NK * NV, triton.cdiv(T, BTL), B * H)

        assert NK == 1, "will encounter some synchronization issue if not."

        o = torch.empty(NK, B, H, T, V, device=q.device)
        z = torch.empty(NK, B, H, T, device=q.device)
        parallel_rebased_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            z,
            q.stride(1),
            q.stride(2),
            q.stride(3),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            scale,
            B=B,
            H=H,
            T=T,
            K=K,
            V=V,
            BTL=BTL,
            BTS=BTS,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        return o.sum(0).to(q.dtype), z.sum(0).to(q.dtype)

    @staticmethod
    def backward(ctx, do, dz):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        BTL, BTS = 64, 32
        assert BTL % BTS == 0
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        BK, BV = max(BK, 16), max(BV, 16)
        B, H, T, K, V = *k.shape, v.shape[-1]
        num_stages = 2
        num_warps = 4
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        grid = (NK * NV, triton.cdiv(T, BTL), B * H)

        assert NK == 1, "will encounter some synchronization issue if not"

        dq = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=q.device)
        dk = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=q.device)
        dv = torch.empty(NK, B, H, T, V, dtype=q.dtype, device=q.device)

        parallel_rebased_bwd_kernel[grid](
            q,
            k,
            v,
            do,
            dz,
            dq,
            dk,
            dv,
            q.stride(1),
            q.stride(2),
            q.stride(3),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            scale,
            B=B,
            H=H,
            T=T,
            K=K,
            V=V,
            BTL=BTL,
            BTS=BTS,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.sum(0).to(v.dtype), None


triton_parallel_based = ParallelBasedFunction.apply


def parallel_rebased(
    q, k, v, eps=1e-5, use_scale=True, use_normalize=True, return_both=False
):
    assert q.shape[-1] <= 128, "only support feature dim up to 128"
    if use_scale:
        scale = q.shape[-1] ** -0.5
    else:
        scale = 1
    o, z = triton_parallel_based(q, k, v, scale)
    if return_both:
        return o, z
    if use_normalize:
        o = o / (z[..., None] + eps)
    else:
        o = o
    return o.to(q.dtype)


import torch


def test_parallel_rebased_with_backward():

    B, H, T, D_head_K, D_head_V = 2, 4, 128, 64, 64

    q = torch.randn(
        B, H, T, D_head_K, device="cuda", dtype=torch.float32, requires_grad=True
    )
    k = torch.randn(
        B, H, T, D_head_K, device="cuda", dtype=torch.float32, requires_grad=True
    )
    v = torch.randn(
        B, H, T, D_head_V, device="cuda", dtype=torch.float32, requires_grad=True
    )

    results = {}

    output = parallel_rebased(
        q, k, v, eps=1e-5, use_scale=True, use_normalize=True, return_both=False
    )
    loss = output.sum()
    loss.backward()
    results["test_case_1"] = {
        "output_shape": output.shape,
        "loss": loss.item(),
        "grad_q": q.grad.norm().item(),
        "grad_k": k.grad.norm().item(),
        "grad_v": v.grad.norm().item(),
    }

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    output = parallel_rebased(
        q, k, v, eps=1e-5, use_scale=True, use_normalize=False, return_both=False
    )
    loss = output.sum()
    loss.backward()
    results["test_case_2"] = {
        "output_shape": output.shape,
        "loss": loss.item(),
        "grad_q": q.grad.norm().item(),
        "grad_k": k.grad.norm().item(),
        "grad_v": v.grad.norm().item(),
    }

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    output, normalizer = parallel_rebased(
        q, k, v, eps=1e-5, use_scale=True, use_normalize=True, return_both=True
    )
    loss = output.sum() + normalizer.sum()
    loss.backward()
    results["test_case_3"] = {
        "output_shape": output.shape,
        "normalizer_shape": normalizer.shape,
        "loss": loss.item(),
        "grad_q": q.grad.norm().item(),
        "grad_k": k.grad.norm().item(),
        "grad_v": v.grad.norm().item(),
    }

    return results


result_gold = test_parallel_rebased_with_backward()
