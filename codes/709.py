import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    A_ptr, B_ptr, C_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):

    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    c = a + b

    tl.store(C_ptr + offsets, c, mask=mask)


def vector_add_triton(
    A: torch.Tensor, B: torch.Tensor, BLOCK_SIZE: int = 1024
) -> torch.Tensor:

    assert (
        A.numel() == B.numel()
    ), "Input vectors must have the same number of elements."
    n_elements = A.numel()

    C = torch.empty_like(A)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](A, B, C, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return C


if __name__ == "__main__":

    n = 1024 * 10
    A = torch.arange(0, n, device="cuda", dtype=torch.float32)
    B = torch.arange(n, 2 * n, device="cuda", dtype=torch.float32)

    C_triton = vector_add_triton(A, B)

    C_pytorch = A + B

    if torch.allclose(C_triton, C_pytorch):
        print("Success: The Triton kernel result matches the PyTorch result!")
    else:
        print("Error: The results do not match.")

    print("Result (first 10 elements):", C_triton[:10])
