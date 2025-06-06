import time
import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)

    y = tl.maximum(x, 0.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def relu_triton(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:

    N = x.numel()

    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    relu_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


def benchmark(func, *args, n_warmup=10, n_iters=100):

    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms


if __name__ == "__main__":

    N = 1024 * 1024
    x = torch.randn(N, device="cuda", dtype=torch.float32)

    y_triton = relu_triton(x)

    y_torch = torch.relu(x)

    if torch.allclose(y_triton, y_torch):
        print("Success: Triton ReLU matches PyTorch ReLU!")
    else:
        print("Error: The Triton ReLU output does not match PyTorch.")

    triton_time = benchmark(relu_triton, x)
    print(f"Average execution time (Triton ReLU): {triton_time:.3f} ms")

    torch_time = benchmark(torch.relu, x)
    print(f"Average execution time (PyTorch ReLU): {torch_time:.3f} ms")
