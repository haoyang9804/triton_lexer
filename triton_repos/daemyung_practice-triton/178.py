import torch
import triton
import triton.language as tl
import triton.testing as testing


@triton.jit
def argmax_kernel(output_ptr, input_ptr, num_batches, size, block_size: tl.constexpr):
    batch = tl.program_id(0)

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(num_batches,),
        strides=(1,),
        offsets=(batch,),
        block_shape=(1,),
        order=(0,),
    )
    input_block_ptr = tl.make_block_ptr(
        input_ptr,
        shape=(num_batches, size),
        strides=(size, 1),
        offsets=(batch, 0),
        block_shape=(1, block_size),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr, boundary_check=(1,))
    condition = tl.arange(0, block_size) < size
    input = tl.where(condition, input, float("-inf"))
    output = tl.argmax(input, 1)
    tl.store(output_block_ptr, output.to(tl.int64))


def argmax(input, dim):
    if dim != 1:
        raise RuntimeError("Only 1 dim is supported.")

    num_batches, size = input.shape
    output = torch.empty(num_batches, device=input.device, dtype=torch.int64)
    block_size = triton.next_power_of_2(size)

    def grid(meta):
        return (num_batches,)

    argmax_kernel[grid](output, input, num_batches, size, block_size)

    return output


def validate():
    input = torch.rand(2, 4096, device="cuda")
    assert torch.allclose(argmax(input, 1), torch.argmax(input, 1))


@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["size"],
            x_vals=[256 * i for i in range(1, 11, 1)],
            x_log=True,
            line_arg="backend",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            ylabel="milliseconds",
            plot_name="argmax-performance",
            args={"num_batches": 8},
        ),
    ]
)
def benchmark(num_batches, size, backend):
    input = torch.rand(num_batches, size, device="cuda")

    if backend == "triton":
        return testing.do_bench(lambda: argmax(input, 1))
    else:
        return testing.do_bench(lambda: torch.argmax(input, 1))


def main():
    validate()
    benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    main()
