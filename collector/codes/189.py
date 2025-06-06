import triton
import triton.language as tl
import torch

from triton.language.extra import libdevice

triton_tanh = libdevice.tanh
triton_cast = tl.cast
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
def _exact_forward_kernel(
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

    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)

    f_row = f_row.to(g_row.dtype)

    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


def geglu_exact_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device="cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _exact_forward_kernel[grid](
        gate,
        up,
        out,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return out


@triton.jit
def _exact_backward_kernel(
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

    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)

    f_row = f_partial_row * e_row
    f_row = f_row.to(dY_row.dtype)

    dg_row = dY_row * f_row

    t = 0.3989422804014327
    df_de = f_partial_row + t * e_row * tl.exp(-0.5 * e_row * e_row)

    de_row = g_row.to(tl.float32) * df_de
    de_row = de_row.to(dY_row.dtype) * dY_row

    tl.store(e + offsets, de_row, mask=mask)
    tl.store(g + offsets, dg_row, mask=mask)


def geglu_exact_backward_kernel(DW, e, g):
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _exact_backward_kernel[grid](
        DW,
        e,
        g,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return e, g


@triton.jit
def _approx_forward_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s = 0.7978845608028654

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    inner_term = s * e_row * (1.0 + 0.044715 * e_row * e_row)
    f_row = 0.5 * e_row * (triton_tanh(inner_term) + 1.0)

    f_row = f_row.to(g_row.dtype)

    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


def geglu_approx_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    n_elements = gate.numel()
    out = torch.empty((batch, seq_len, hd), dtype=gate.dtype, device="cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _approx_forward_kernel[grid](
        gate,
        up,
        out,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return out


@triton.jit
def _approx_backward_kernel(
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

    a = 0.7978845608028654
    b = 0.044715 * a

    inner = e_row * (a + b * e_row * e_row)

    tanh_inner = triton_tanh(inner)

    v = 1.0 + tanh_inner

    f_row = 0.5 * e_row * v

    df_de = 0.5 * v * (1.0 + e_row * (2.0 - v) * (a + 3.0 * b * e_row * e_row))

    dg_row = dY_row * f_row

    de_row = g_row * df_de
    de_row = de_row.to(dY_row.dtype) * dY_row

    tl.store(e + offsets, de_row, mask=mask)
    tl.store(g + offsets, dg_row, mask=mask)


def geglu_approx_backward_kernel(dY, e, g):
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _approx_backward_kernel[grid](
        dY,
        e,
        g,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return e, g


def test_geglu_correctness(use_approx=False):

    import torch
    import torch.nn.functional as F

    def geglu_reference_forward(x):

        x_chunks = torch.chunk(x, 2, dim=-1)
        gate, value = x_chunks[0], x_chunks[1]
        return value * F.gelu(gate)

    forward_kernel = _approx_forward_kernel if use_approx else _exact_forward_kernel
    backward_kernel = _approx_backward_kernel if use_approx else _exact_backward_kernel

    implementation_type = "approximate" if use_approx else "exact"
    print(f"Testing {implementation_type} GEGLU implementation...")

    def test_forward():

        print(f"Testing {implementation_type} GEGLU forward pass...")

        batch_size, seq_len, hidden_dim = 2, 10, 128
        x = torch.randn(
            batch_size, seq_len, hidden_dim * 2, device="cuda", requires_grad=True
        )

        ref_output = geglu_reference_forward(x)

        x_chunks = torch.chunk(x, 2, dim=-1)
        gate, value = x_chunks[0], x_chunks[1]
        gate_flat = gate.reshape(-1)
        value_flat = value.reshape(-1)

        output_flat = torch.empty_like(gate_flat)

        n_elements = gate_flat.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        forward_kernel[grid](
            gate_flat, value_flat, output_flat, n_elements, BLOCK_SIZE=1024
        )

        our_output = output_flat.reshape(gate.shape)

        max_diff = torch.max(torch.abs(ref_output - our_output))
        print(
            f"Max difference in {implementation_type} GEGLU forward pass: {max_diff.item()}"
        )
        assert (
            max_diff < 1e-2 if use_approx else 1e-5
        ), f"{implementation_type} GEGLU forward pass implementation is incorrect!"
        return True

    def test_backward():

        print(f"Testing {implementation_type} GEGLU backward pass...")

        batch_size, seq_len, hidden_dim = 2, 10, 128
        x = torch.randn(
            batch_size, seq_len, hidden_dim * 2, device="cuda", requires_grad=True
        )

        x_ref = x.clone().detach().requires_grad_(True)
        ref_output = geglu_reference_forward(x_ref)

        grad_output = torch.randn_like(ref_output)

        ref_output.backward(grad_output)
        ref_grad = x_ref.grad.clone()

        x_chunks = torch.chunk(x, 2, dim=-1)
        gate, value = x_chunks[0], x_chunks[1]
        gate_flat = gate.reshape(-1)
        value_flat = value.reshape(-1)

        output_flat = torch.empty_like(gate_flat)
        n_elements = gate_flat.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        forward_kernel[grid](
            gate_flat, value_flat, output_flat, n_elements, BLOCK_SIZE=1024
        )

        grad_output_flat = grad_output.reshape(-1)

        dW = grad_output_flat.clone()
        e = gate_flat.clone()
        g = value_flat.clone()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        backward_kernel[grid](dW, e, g, n_elements, BLOCK_SIZE=1024)

        our_grad = torch.cat([e.reshape(gate.shape), g.reshape(value.shape)], dim=-1)

        max_diff = torch.max(torch.abs(ref_grad - our_grad))
        print(
            f"Max difference in {implementation_type} GEGLU backward pass: {max_diff.item()}"
        )
        assert (
            max_diff < 1e-2 if use_approx else 1e-5
        ), f"{implementation_type} GEGLU backward pass implementation is incorrect!"
        return True

    forward_passed = test_forward()
    backward_passed = test_backward()

    if forward_passed and backward_passed:
        print(
            f"All tests passed! {implementation_type.capitalize()} GEGLU implementation is correct."
        )
    else:
        print(
            f"Tests failed! {implementation_type.capitalize()} GEGLU implementation needs fixing."
        )


if __name__ == "__main__":

    test_geglu_correctness(use_approx=False)

    test_geglu_correctness(use_approx=True)
