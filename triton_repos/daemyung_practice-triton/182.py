import triton
import triton.language as tl


@triton.jit
def print_grid():
    pid = tl.program_id(0)
    tl.device_print("pid: ", pid)


def main():
    def grid(meta):
        return (2,)

    print_grid[grid]()


if __name__ == "__main__":
    main()
