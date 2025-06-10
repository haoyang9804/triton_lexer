import torch
import triton
import util

import trident


@util.report(
    "instance norm forward",
    ["x_size"],
    [128 * i for i in range(1, 21)],
    {"num_batches": 64, "y_size": 32},
)
def bench_instance_norm_forward(num_batches, y_size, x_size, dtype, backend):
    input = torch.randn(num_batches, y_size, x_size, device="cuda", dtype=dtype)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.instance_norm(input)
        )
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.instance_norm(input)
        )


@util.report(
    "instance norm backward",
    ["x_size"],
    [128 * i for i in range(1, 21)],
    {"num_batches": 64, "y_size": 32},
)
def bench_instance_norm_backward(num_batches, y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(
        num_batches, y_size, x_size, **factory_kwargs, requires_grad=True
    )
    grad_output = torch.rand_like(input)

    if backend == "torch":
        output = torch.nn.functional.instance_norm(input)
    else:
        output = trident.function.instance_norm(input)

    return triton.testing.do_bench_cudagraph(
        lambda: output.backward(grad_output, retain_graph=True)
    )


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_instance_norm_forward.run(
            print_data=True, show_plots=show_plots, dtype=dtype
        )
    else:
        bench_instance_norm_backward.run(
            print_data=True, show_plots=show_plots, dtype=dtype
        )
