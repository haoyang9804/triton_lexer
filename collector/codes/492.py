import torch
import triton
import util

import trident


@util.report(
    "max forward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 256}
)
def bench_max_forward(y_size, x_size, dtype, backend):
    input = torch.randn(y_size, x_size, device="cuda", dtype=dtype)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.max(input, 1))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.max(input, 1))


@util.report(
    "max backward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 256}
)
def bench_max_backward(y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.randn(y_size, **factory_kwargs)

    if backend == "torch":
        output, _ = torch.max(input, 1)
    else:
        output, _ = trident.function.max(input, 1)

    return triton.testing.do_bench_cudagraph(
        lambda: output.backward(grad_output, retain_graph=True)
    )


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_max_forward.run(print_data=True, show_plots=show_plots, dtype=dtype)
    else:
        bench_max_backward.run(print_data=True, show_plots=show_plots, dtype=dtype)
