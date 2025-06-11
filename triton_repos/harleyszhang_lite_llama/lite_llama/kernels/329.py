import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_update_kv_index(
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    select_index,
    stride_req_to_token_b,
    stride_req_to_token_s,
):

    cur_index = tl.program_id(0)

    cur_req_idx = tl.load(b_req_idx + cur_index)

    cur_token_index = tl.load(select_index + cur_index)

    cur_seq_len = tl.load(b_seq_len + cur_index)

    dest_offset = (
        req_to_token_indexs
        + cur_req_idx * stride_req_to_token_b
        + (cur_seq_len - 1) * stride_req_to_token_s
    )

    tl.store(dest_offset, cur_token_index)

    return


@torch.no_grad()
def update_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, select_index):

    seq_len = b_seq_len.shape[0]

    assert (
        b_seq_len.shape[0] == select_index.shape[0]
        and b_req_idx.shape[0] == b_seq_len.shape[0]
    ), "所有输入张量在第一个维度上的大小必须相同。"

    grid = (seq_len,)

    num_warps = 1

    _fwd_kernel_update_kv_index[grid](
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        select_index,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        num_warps=num_warps,
        num_stages=1,
    )
    return
