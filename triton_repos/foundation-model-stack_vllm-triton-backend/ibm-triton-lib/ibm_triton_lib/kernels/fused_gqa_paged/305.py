import triton.language as tl
import triton





def _generate_asm(num_pack):
    template = 
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_constraints(num_pack):
    return (
        ",".join("=r" for i in range(num_pack))
        + ","
        + ",".join("r" for i in range(num_pack))
    )


NUM_REG: tl.constexpr = 1
asm_str: tl.constexpr = _generate_asm(NUM_REG)
constraints_str: tl.constexpr = _generate_constraints(NUM_REG)


@triton.jit
def softplus(x, is_compiling: tl.constexpr = False):
    if is_compiling:
        tl.static_print("Using triton softplus.")
        out = tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)
        return out
    else:
        out = tl.inline_asm_elementwise(
            asm=asm_str,
            constraints=constraints_str,
            pack=NUM_REG,
            args=[
                x,
            ],
            dtype=tl.float32,
            is_pure=True,
        )
        return out


@triton.jit
def cumsum(x, block_range=None, USE_DOT_CUMSUM: tl.constexpr = False):
    if USE_DOT_CUMSUM:
        cm = tl.where(
            block_range[:, None] >= block_range[None, :], 1.0, 0.0
        )  
        return tl.dot(x, cm)
    else:
        return tl.cumsum(x, axis=1, reverse=True)


@triton.jit
def get_split_tblocks_range(split_idx, kv_len, BLOCK_T, num_splits):
    num_tblocks = (kv_len + BLOCK_T - 1) // BLOCK_T
    tblock_start = (split_idx * num_tblocks) // num_splits
    tblock_end = ((split_idx + 1) * num_tblocks) // num_splits
    return tblock_start, tblock_end


@triton.jit
def attend_one_block(
    q,
    k,
    v,
    qk_scale,
    m_i,
    d_i,
    acc,
    alibi_slopes,  
    alibi_distances,  
    IS_LAST_BLOCK,  
    tb_len_max,  
    offs_t: tl.constexpr,
    FORCE_FP16_PV: tl.constexpr,
    QUANTIZE_P: tl.constexpr,
    MAX_FP8: tl.constexpr,
    IS_STICKBREAKING: tl.constexpr,
    USE_DOT_CUMSUM: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    ATTEND_CURRENT: tl.constexpr,
):
    kv_len_dim: tl.constexpr = 1 if not TRANSPOSED else 0  

    
    if not TRANSPOSED:
        k = k.T  
        logits = tl.dot(q, k, out_dtype=tl.float32)  
    else:
        q = q.T  
        logits = tl.dot(k, q, out_dtype=tl.float32)  

    logits *= qk_scale  

    if USE_ALIBI_SLOPES:
        alibi_biases = (
            alibi_slopes[:, None] * alibi_distances[None, :]
        )  
        logits += alibi_biases if not TRANSPOSED else alibi_biases.T

    
    t_mask = offs_t < tb_len_max
    if IS_LAST_BLOCK:
        if not IS_STICKBREAKING:
            t_mask_logits = t_mask[None, :] if not TRANSPOSED else t_mask[:, None]
            logits += tl.where(t_mask_logits, 0.0, float("-inf"))
        else:
            
            t_mask = offs_t < (tb_len_max if ATTEND_CURRENT else (tb_len_max - 1))

    if not IS_STICKBREAKING:  
        
        m_i_new = tl.maximum(
            m_i, tl.max(logits, axis=kv_len_dim)
        )  

        alpha = tl.math.exp2(m_i - m_i_new)  
        p = tl.math.exp2(
            logits
            - (
                m_i_new[:, None]  
                if not TRANSPOSED
                else m_i_new[None, :]
            )
        )

        
        acc *= alpha[:, None] if not TRANSPOSED else alpha[None, :]  
        
        m_i = m_i_new  
        d_i = d_i * alpha + tl.sum(p, axis=kv_len_dim)  
    else:  
        
        log_om_beta = -softplus(
            logits,
        )  

        if TRANSPOSED:
            log_om_beta = log_om_beta.T  
            logits = logits.T

        if IS_LAST_BLOCK:  
            log_om_beta = tl.where(t_mask[None, :], log_om_beta, 0.0)

        log_p = logits + d_i[:, None]  
        d_i += tl.sum(log_om_beta, axis=1)  
        log_p += cumsum(log_om_beta, block_range=offs_t, USE_DOT_CUMSUM=USE_DOT_CUMSUM)

        
        p = tl.math.exp2(log_p)  

        if IS_LAST_BLOCK:  
            p = tl.where(t_mask[None, :], p, 0.0)  

        if TRANSPOSED:
            p = p.T  

    p_scale = 1.0
    if FORCE_FP16_PV:
        
        v = v.to(tl.float16)
    else:
        
        if QUANTIZE_P and v.dtype == tl.float8e4nv:
            tl.static_assert(
                not IS_STICKBREAKING
            )  
            
            p_max = tl.max(tl.abs(p), axis=kv_len_dim, keep_dims=True)
            p_scale = p_max / MAX_FP8
            p_invs_scale = 1.0 / p_scale
            p = p * p_invs_scale  
    p = p.to(v.dtype)

    if not TRANSPOSED:
        acc += tl.dot(p, v, out_dtype=tl.float32) * p_scale  
    else:
        acc += tl.dot(v.T, p, out_dtype=tl.float32) * p_scale  

    return m_i, d_i, acc
