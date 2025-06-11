import triton
import triton.language as tl


@triton.jit
def hello_triton():
    tl.device_print("Hello Triton!")


def main():
    hello_triton[(1,)]()


if __name__ == "__main__":
    main()
