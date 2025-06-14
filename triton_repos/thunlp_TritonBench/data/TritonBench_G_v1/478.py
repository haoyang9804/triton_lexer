import torch
import triton
import triton.language as tl


@triton.jit
def attention_fwd_kernel(
    q,
    k,
    v,
    h,
    o,
    s_qh,
    s_qt,
    s_qd,
    s_hh,
    s_ht,
    T,
    scale,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NT: tl.constexpr,
    STORE: tl.constexpr,
    IFCOND: tl.constexpr,
):
    i_bh = tl.program_id(0)

    b_h = tl.zeros([BD, BD], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(
            q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i * BT), (BD, BT), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0)
        )
        p_h = tl.make_block_ptr(
            h + i_bh * s_hh, (NT * BD, BD), (s_ht, s_qd), (i * BD, 0), (BD, BD), (1, 0)
        )
        p_o = tl.make_block_ptr(
            o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0)
        )

        if STORE:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty))

        b_q = tl.load(p_q)
        b_q = (b_q * scale).to(b_q.dtype)

        b_k = tl.load(p_k)

        b_v = tl.load(p_v)

        b_s = tl.dot(b_q, b_k, allow_tf32=False)

        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        if IFCOND:
            if i == 0:
                b_h = tl.dot(b_k, b_v, allow_tf32=False)
            else:
                b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
                b_h += tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h += tl.dot(b_k, b_v, allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty))


class AttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, store=False, ifcond=False):
        batch_size, n_heads, seq_len, d_head = q.shape
        scale = d_head**-0.5
        BD = q.shape[-1]
        BT = 32
        NT = triton.cdiv(seq_len, BT)
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4

        h = q.new_empty(batch_size, n_heads, NT * BD, BD)
        o = torch.empty_like(q)
        grid = (batch_size * n_heads,)
        attention_fwd_kernel[grid](
            q,
            k,
            v,
            h,
            o,
            q.stride(1),
            q.stride(2),
            q.stride(3),
            h.stride(1),
            h.stride(2),
            seq_len,
            scale,
            BT=BT,
            BD=BD,
            NT=NT,
            STORE=store,
            IFCOND=ifcond,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return o


import torch


def test_attention_fwd():

    B, H, T, D = 2, 8, 1024, 128
    q = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")
    k = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")
    v = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")

    result = AttentionFunction.apply(q, k, v)

    result_store = AttentionFunction.apply(q, k, v, True)

    result_ifcond = AttentionFunction.apply(q, k, v, False, True)

    result_store_ifcond = AttentionFunction.apply(q, k, v, True, True)

    B, H, T, D = 2, 8, 1024, 128
    q_large_head = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")
    k_large_head = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")
    v_large_head = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")

    result_large_head = AttentionFunction.apply(
        q_large_head, k_large_head, v_large_head
    )

    B, H, T, D = 2, 8, 1, 128
    q_small_seq = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")
    k_small_seq = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")
    v_small_seq = torch.randn((B, H, T, D), dtype=torch.float, device="cuda")

    result_small_seq = AttentionFunction.apply(q_small_seq, k_small_seq, v_small_seq)

    return {
        "test_case_1": result,
        "test_case_2": result_store,
        "test_case_3": result_ifcond,
        "test_case_4": result_store_ifcond,
        "test_case_5": result_large_head,
        "test_case_6": result_small_seq,
    }


result_gold = test_attention_fwd()
