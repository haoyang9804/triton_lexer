import torch
import triton
import triton.language as tl


@triton.jit
def diag_ssm_forward_kernel(
    s_ptr, x_ptr, lambda_ptr, y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr
):

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim
    s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)
    for t in range(length):
        offsets = t * batch_size * dim + col_offsets
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        s = s * Lambda + x
        tl.store(y_ptr + offsets, s, mask=mask)


@triton.jit
def diag_ssm_backward_kernel(
    s_ptr,
    lambda_ptr,
    y_ptr,
    grad_s_ptr,
    grad_x_ptr,
    grad_lambda_ptr,
    grad_y_ptr,
    length,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)

    grad_s = tl.zeros_like(Lambda)
    grad_Lambda = tl.zeros_like(Lambda)

    for i in range(length):

        t = length - 1 - i
        offsets = t * batch_size * dim + col_offsets

        grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        if t > 0:
            s = tl.load(y_ptr + offsets - batch_size * dim, mask=mask, other=0)
        else:
            s = tl.load(s_ptr + col_offsets, mask=mask, other=0)

        grad_s = grad_y + grad_s
        grad_x = grad_s
        grad_Lambda += grad_s * s
        grad_s = grad_s * Lambda

        tl.store(grad_x_ptr + offsets, grad_x, mask=mask)

    tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)


@triton.jit
def diag_ssm_forward_kernel_complex(
    s_ptr, x_ptr, y_ptr, lambda_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr
):

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    s_real = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
    s_imag = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
    lambda_real = tl.load(lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    for t in range(length):
        offsets = (t * batch_size * dim + col_offsets) * 2

        x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
        x_imag = tl.load(x_ptr + offsets + 1, mask=mask, other=0)

        new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
        new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag

        tl.store(y_ptr + offsets, new_s_real, mask=mask)
        tl.store(y_ptr + offsets + 1, new_s_imag, mask=mask)

        s_real, s_imag = new_s_real, new_s_imag


@triton.jit
def diag_ssm_backward_kernel_complex(
    s_ptr,
    lambda_ptr,
    y_ptr,
    grad_s_ptr,
    grad_x_ptr,
    grad_lambda_ptr,
    grad_y_ptr,
    length,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    lambda_real = tl.load(lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    grad_s_real = tl.zeros_like(lambda_real)
    grad_s_imag = tl.zeros_like(lambda_imag)
    grad_lambda_real = tl.zeros_like(lambda_real)
    grad_lambda_imag = tl.zeros_like(lambda_imag)

    for i in range(length):

        t = length - 1 - i
        offsets = (t * batch_size * dim + col_offsets) * 2

        grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        grad_y_imag = -tl.load(grad_y_ptr + offsets + 1, mask=mask, other=0)
        if t > 0:
            s_real = tl.load(y_ptr + offsets - 2 * batch_size * dim, mask=mask, other=0)
            s_imag = tl.load(
                y_ptr + offsets - 2 * batch_size * dim + 1, mask=mask, other=0
            )
        else:
            s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
            s_imag = tl.load(s_ptr + 2 * col_offsets + 1, mask=mask, other=0)

        grad_s_real = grad_y_real + grad_s_real
        grad_s_imag = grad_y_imag + grad_s_imag
        grad_x_real = grad_s_real
        grad_x_imag = grad_s_imag
        grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
        grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
        grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
        grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real

        tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
        tl.store(grad_x_ptr + offsets + 1, -grad_x_imag, mask=mask)

    tl.store(grad_s_ptr + col_offsets * 2, grad_s_real, mask=mask)
    tl.store(grad_s_ptr + col_offsets * 2 + 1, -grad_s_imag, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets * 2, grad_lambda_real, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets * 2 + 1, -grad_lambda_imag, mask=mask)


class _ssm_forward(torch.autograd.Function):

    BLOCK_SIZE = 128

    @staticmethod
    def forward(ctx, s, x, Lambda):
        assert s.is_contiguous() and x.is_contiguous() and Lambda.is_contiguous()
        length, batch_size, dim = x.shape
        n = batch_size * dim
        y = torch.zeros_like(x)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        if Lambda.dtype == torch.complex64:

            if not torch.is_complex(s):
                raise ValueError("当Lambda为复数时，s必须是复数张量")
            if not torch.is_complex(x):
                raise ValueError("当Lambda为复数时，x必须是复数张量")
            diag_ssm_forward_kernel_complex[grid](
                torch.view_as_real(s),
                torch.view_as_real(x),
                torch.view_as_real(y),
                torch.view_as_real(Lambda),
                length,
                batch_size,
                dim,
                _ssm_forward.BLOCK_SIZE,
            )
        elif Lambda.dtype.is_floating_point:
            diag_ssm_forward_kernel[grid](
                s, x, Lambda, y, length, batch_size, dim, _ssm_forward.BLOCK_SIZE
            )
        else:
            raise ValueError("不支持的 dtype: %s" % Lambda.dtype)
        ctx.save_for_backward(s, y, Lambda)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        s, y, Lambda = ctx.saved_tensors
        length, batch_size, dim = y.shape
        grad_y = grad_y.contiguous()
        n = batch_size * dim
        grad_s = torch.empty_like(s)
        grad_x = torch.empty_like(grad_y)

        grad_lambda = torch.empty_like(s)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        if Lambda.dtype == torch.complex64:
            diag_ssm_backward_kernel_complex[grid](
                torch.view_as_real(s),
                torch.view_as_real(Lambda),
                torch.view_as_real(y),
                torch.view_as_real(grad_s),
                torch.view_as_real(grad_x),
                torch.view_as_real(grad_lambda),
                torch.view_as_real(grad_y),
                length,
                batch_size,
                dim,
                _ssm_forward.BLOCK_SIZE,
            )
        else:
            diag_ssm_backward_kernel[grid](
                s,
                Lambda,
                y,
                grad_s,
                grad_x,
                grad_lambda,
                grad_y,
                length,
                batch_size,
                dim,
                _ssm_forward.BLOCK_SIZE,
            )
        return grad_s, grad_x, grad_lambda.sum(dim=0)


diag_ssm_forward_triton = _ssm_forward.apply


def test_diag_ssm_triton():

    batch_size, dim, length = 2, 3, 5
    BLOCK_SIZE = 128

    s_real = torch.randn(
        (batch_size, dim), dtype=torch.float32, device="cuda", requires_grad=True
    )
    x_real = torch.randn(
        (length, batch_size, dim),
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    Lambda_real = torch.rand(
        (dim,), dtype=torch.float32, device="cuda", requires_grad=True
    )

    s_complex = torch.randn(
        (batch_size, dim), dtype=torch.complex64, device="cuda", requires_grad=True
    )
    x_complex = torch.randn(
        (length, batch_size, dim),
        dtype=torch.complex64,
        device="cuda",
        requires_grad=True,
    )
    Lambda_complex = torch.rand(
        (dim,), dtype=torch.complex64, device="cuda", requires_grad=True
    )

    y_triton_real = diag_ssm_forward_triton(s_real, x_real, Lambda_real)

    y_triton_complex = diag_ssm_forward_triton(s_complex, x_complex, Lambda_complex)

    grad_output_real = torch.ones_like(y_triton_real, device="cuda")
    y_triton_real.backward(grad_output_real)

    grad_output_complex = torch.ones_like(y_triton_complex, device="cuda")
    y_triton_complex.backward(grad_output_complex)

    results = {
        "test_case_1": {
            "y_triton_real": y_triton_real,
            "grad_s_real": s_real.grad.clone(),
            "grad_x_real": x_real.grad.clone(),
            "grad_Lambda_real": Lambda_real.grad.clone(),
        },
        "test_case_2": {
            "y_triton_complex": y_triton_complex,
            "grad_s_complex": s_complex.grad.clone(),
            "grad_x_complex": x_complex.grad.clone(),
            "grad_Lambda_complex": Lambda_complex.grad.clone(),
        },
    }

    return results


if __name__ == "__main__":
    result_gold = test_diag_ssm_triton()

    for test_case, outputs in result_gold.items():
        print(f"{test_case}:")
        for name, tensor in outputs.items():
            print(f"  {name}: {tensor}")
