import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n: int) -> (
    int,
    int,
):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _forward_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)

    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


def swiglu_forward_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _forward_kernel[grid](
        e,
        g,
        h,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return h


@triton.jit
def _backward_kernel(
    dY,
    e,
    g,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dY_row = tl.load(dY + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    se_row = tl.sigmoid(e_row)

    f_row = se_row * e_row
    f_row = f_row.to(dY_row.dtype)

    dg_row = dY_row * f_row

    de_row = (
        dY_row.to(tl.float32)
        * g_row.to(tl.float32)
        * se_row
        * (1.0 + e_row * (1.0 - se_row))
    )
    de_row = de_row.to(dY_row.dtype)

    tl.store(e + offsets, de_row, mask=mask)
    tl.store(g + offsets, dg_row, mask=mask)


def swiglu_DWf_DW_dfg_kernel(dY, e, g):
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _backward_kernel[grid](
        dY,
        e,
        g,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return e, g


def test_swiglu_correctness():

    import torch
    import torch.nn.functional as F

    def swiglu_reference_forward(e, g):

        return g * (e * F.sigmoid(e))

    forward_kernel = _forward_kernel
    backward_kernel = _backward_kernel

    batch_size, seq_len, hidden_dim = 2, 10, 128
    e = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device="cuda")
    g = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device="cuda")

    h = torch.empty_like(e)

    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    forward_kernel[grid](e, g, h, n_elements, BLOCK_SIZE=1024)

    our_output = h.clone()

    ref_output = swiglu_reference_forward(e, g)

    max_diff = torch.max(torch.abs(ref_output - our_output))
    print(f"Max difference in SwiGLU forward pass: {max_diff.item()}")
    assert max_diff < 1e-5, "SwiGLU forward pass implementation is incorrect!"

    dY = torch.randn_like(h)

    e.requires_grad_(True)
    g.requires_grad_(True)
    ref_output = swiglu_reference_forward(e, g)
    ref_output.backward(dY)
    ref_de = e.grad.clone()
    ref_dg = g.grad.clone()

    backward_kernel[grid](dY, e, g, n_elements, BLOCK_SIZE=1024)

    max_diff_de = torch.max(torch.abs(ref_de - e))
    print(f"Max difference in SwiGLU backward pass (de): {max_diff_de.item()}")
    assert (
        max_diff_de < 1e-5
    ), "SwiGLU backward pass implementation for de is incorrect!"

    max_diff_dg = torch.max(torch.abs(ref_dg - g))
    print(f"Max difference in SwiGLU backward pass (dg): {max_diff_dg.item()}")
    assert (
        max_diff_dg < 1e-5
    ), "SwiGLU backward pass implementation for dg is incorrect!"

    print("All tests passed!")


if __name__ == "__main__":
    test_swiglu_correctness()
