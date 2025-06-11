import torch
import triton
import triton.language as tl


@triton.jit
def atomic_kernel(x_ptr, increment):
    tl.atomic_add(x_ptr, increment)


def atomic(increment):
    x = torch.zeros(1, device="cuda")

    def grid(meta):
        return (1024,)

    atomic_kernel[grid](x, increment)

    return x


def main():
    x = atomic(2)
    print(x)


if __name__ == "__main__":
    main()
