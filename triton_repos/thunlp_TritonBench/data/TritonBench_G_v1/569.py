import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):

    off_bh = tl.program_id(0)
    off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    e_offset = off_e * BLOCK_MODEL

    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]

    off_block = tl.arange(0, BLOCK)

    index = off_block[:, None] - off_block[None, :]
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    for i in range(NUM_BLOCK):

        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        qk = tl.dot(q, k_trans)
        qk = tl.where(index >= 0, qk, 0)
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter

        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv += tl.dot(k_trans, v)
        off_block += BLOCK


@triton.jit
def _bwd_intra_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):

    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK + tl.arange(0, BLOCK)

    Q_trans_block_ptr = (
        Q + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    K_block_ptr = K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = (
        V + v_offset + block_offset[None, :] * e + tl.arange(0, e)[:, None]
    )

    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    DK_trans_block_ptr = (
        DK + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]

    array = tl.arange(0, BLOCK).to(tl.float32)

    index = array[:, None] - array[None, :]

    k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )
    do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )

    dqk = tl.dot(do, v_trans)
    dqk = tl.where(index >= 0, dqk, 0)
    dq_intra = tl.dot(dqk, k)

    dk_intra_trans = tl.dot(q_trans, dqk)

    qk_trans = tl.dot(k, q_trans)
    qk_trans = tl.where(index <= 0, qk_trans, 0)
    dv_intra = tl.dot(qk_trans, do)

    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra

    tl.store(
        DQ_block_ptr,
        dq.to(DQ_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )
    tl.store(
        DK_trans_block_ptr,
        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
        mask=block_offset[None, :] < n,
    )
    tl.store(
        DV_block_ptr,
        dv.to(DV_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )


@triton.jit
def _bwd_inter_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):

    off_bh = tl.program_id(0)
    off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    DQ_block_ptr = (
        DQ + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )

    off_block1 = tl.arange(0, CBLOCK)
    off_block2 = tl.arange(0, CBLOCK)

    kv_trans = tl.zeros([e, d], dtype=tl.float32)

    for i in range(NUM_BLOCK):

        for j in range(NUM_CBLOCK):
            if i > 0:
                do = tl.load(DO_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                dq_inter = tl.dot(do, kv_trans)
                dq = dq_inter + tl.load(
                    DQ_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )
                tl.store(
                    DQ_block_ptr,
                    dq.to(DQ_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
            off_block1 += CBLOCK

        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(
                V_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            kv_trans_current += tl.dot(v_trans, k)

            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
            off_block2 += CBLOCK

        kv_trans += kv_trans_current

    m = NUM_BLOCK * BLOCK
    off_block1 = m + tl.arange(0, CBLOCK)
    off_block2 = m + tl.arange(0, CBLOCK)

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    dkv = tl.zeros([d, e], dtype=tl.float32)

    for i in range(NUM_BLOCK - 1, -1, -1):

        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            off_block1 -= CBLOCK

            if i < NUM_BLOCK - 1:
                k = tl.load(K_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                v_trans = tl.load(
                    V_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                ).to(tl.float32)

                dk_inter_trans = tl.dot(dkv, v_trans)
                dv_inter = tl.dot(k, dkv)

                dk_trans = dk_inter_trans + tl.load(
                    DK_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                )
                dv = dv_inter + tl.load(
                    DV_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )

                tl.store(
                    DK_trans_block_ptr,
                    dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
                    mask=off_block1[None, :] < n,
                )
                tl.store(
                    DV_block_ptr,
                    dv.to(DV_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            off_block2 -= CBLOCK

            do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            q_trans = tl.load(
                Q_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            dkv_current += tl.dot(q_trans, do)

        dkv += dkv_current


class LightningAttention2NoDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        BLOCK = 64
        NUM_BLOCK = triton.cdiv(q.shape[2], BLOCK)

        BLOCK_MODEL = min(triton.next_power_of_2(e), 32)
        grid = (b * h, triton.cdiv(e, BLOCK_MODEL))

        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            BLOCK_MODEL=BLOCK_MODEL,
        )

        ctx.save_for_backward(q, k, v)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        do = do.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        b, h, n, d = q.shape
        e = v.shape[-1]

        BLOCK = 64
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK

        grid = (b * h, NUM_BLOCK)
        _bwd_intra_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h,)
        _bwd_inter_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        return dq, dk, dv


lightning_attn2_no_decay = LightningAttention2NoDecay.apply


import torch


def test_lightning_attention2_no_decay():

    b, h, n, d, e = 2, 8, 128, 64, 128

    q = torch.randn(
        (b, h, n, d), dtype=torch.float32, device="cuda", requires_grad=True
    )
    k = torch.randn(
        (b, h, n, d), dtype=torch.float32, device="cuda", requires_grad=True
    )
    v = torch.randn(
        (b, h, n, e), dtype=torch.float32, device="cuda", requires_grad=True
    )

    o = lightning_attn2_no_decay(q, k, v)

    o.sum().backward()

    results = {
        "test_case_1": (
            o.cpu().detach().numpy(),
            q.grad.cpu().detach().numpy(),
            k.grad.cpu().detach().numpy(),
            v.grad.cpu().detach().numpy(),
        )
    }

    return results


result_gold = test_lightning_attention2_no_decay()
print(result_gold)
