import ast

def add_end_marks(source_code):
    class EndMarkTransformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # 处理函数定义
            self.generic_visit(node)
            end_mark = ast.Expr(value=ast.Name(id='enddef', ctx=ast.Load()))
            node.body.append(end_mark)
            return node

        def visit_If(self, node):
            # 处理 if 语句
            self.generic_visit(node)
            
            # 处理 elif 语句
            if node.orelse and isinstance(node.orelse[0], ast.If):
                # 如果是 elif 语句，添加 endif
                end_mark = ast.Expr(value=ast.Name(id='endif', ctx=ast.Load()))
                node.body.append(end_mark)
            else:
                # 如果是 if 语句，添加 endelif
                end_mark = ast.Expr(value=ast.Name(id='endelif', ctx=ast.Load()))
                node.body.append(end_mark)
                
                # 如果有 else 块，在 else 块末尾添加 endelse
                if node.orelse:
                    end_mark = ast.Expr(value=ast.Name(id='endelse', ctx=ast.Load()))
                    node.orelse.append(end_mark)
            return node

        def visit_For(self, node):
            # 处理 for 循环
            self.generic_visit(node)
            end_mark = ast.Expr(value=ast.Name(id='endfor', ctx=ast.Load()))
            node.body.append(end_mark)
            return node

        def visit_While(self, node):
            # 处理 while 循环
            self.generic_visit(node)
            end_mark = ast.Expr(value=ast.Name(id='endwhile', ctx=ast.Load()))
            node.body.append(end_mark)
            return node

        def visit_ClassDef(self, node):
            # 处理类定义
            self.generic_visit(node)
            end_mark = ast.Expr(value=ast.Name(id='endclass', ctx=ast.Load()))
            node.body.append(end_mark)
            return node

        def visit_With(self, node):
            # 处理 with 语句
            self.generic_visit(node)
            end_mark = ast.Expr(value=ast.Name(id='endwith', ctx=ast.Load()))
            node.body.append(end_mark)
            return node

        def visit_Try(self, node):
            # 处理 try 语句
            self.generic_visit(node)
            
            # 在 try 块末尾添加 endtry
            end_mark = ast.Expr(value=ast.Name(id='endtry', ctx=ast.Load()))
            node.body.append(end_mark)
            
            # 处理 except 块
            for handler in node.handlers:
                end_mark = ast.Expr(value=ast.Name(id='endexcept', ctx=ast.Load()))
                handler.body.append(end_mark)
            
            # 处理 finally 块
            if node.finalbody:
                end_mark = ast.Expr(value=ast.Name(id='endfinally', ctx=ast.Load()))
                node.finalbody.append(end_mark)
            
            return node

    # 解析源代码
    tree = ast.parse(source_code)
    
    # 转换 AST
    transformer = EndMarkTransformer()
    modified_tree = transformer.visit(tree)
    
    # 将修改后的 AST 转换回源代码，保持原始格式
    return ast.unparse(modified_tree)

# 测试代码
if __name__ == "__main__":
    test_code = """
def parallel_simple_gla_fwd_kernel(
    q,
    k,
    v,
    g,
    o,
    attn,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    o += i_k * B * T * H * V

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
    if OUTPUT_ATTENTIONS:
        attn += (bos * H + i_h * T) * T + i_k * B * H * T * T
    stride_qk = H * K
    stride_vo = H * V
    stride_g = H

    p_q = tl.make_block_ptr(
        q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)

    o_k = i_t * BT + tl.arange(0, BS)

    if USE_G:
        p_gq = tl.make_block_ptr(g, (T,), (stride_g,), (i_t * BT,), (BT,), (0,))

        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)

    else:
        b_gq = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_qk), (i_k * BK, i_s), (BK, BS), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_vo, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
        )

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))

        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            p_gk = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,))
            b_s *= safe_exp(b_gq[:, None] - b_gk[None, :])
            b_s = tl.where(m_s, b_s, 0)
        else:
            b_s = tl.where(m_s, b_s, 0)

        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_q.dtype), b_v)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(
                attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0)
            )
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        o_k += BS

    for i_s in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_qk), (i_k * BK, i_s), (BK, BS), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_vo, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
        )

        b_k = tl.load(p_k, boundary_check=(0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            p_g = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * stride_g)
            b_gp = tl.load(g + (i_s - 1) * stride_g) if i_s % BT > 0 else 0.0

            b_s *= safe_exp(b_gq[:, None] + (b_gn - b_g)[None, :])
            b_gq += b_gn - b_gp
"""
    modified_code = add_end_marks(test_code)
    print(modified_code)
