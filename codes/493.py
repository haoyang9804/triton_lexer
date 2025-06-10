import torch
import triton
import util

import trident


@util.report(
    "mean forward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 32}
)
def bench_mean_forward(y_size, x_size, dtype, backend):
    input = torch.randn(y_size, x_size, device="cuda", dtype=dtype)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.mean(input, 1))
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.mean(input, 1)
        )


@util.report(
    "mean backward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 32}
)
def bench_mean_backward(y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.empty(y_size, **factory_kwargs)

    if backend == "torch":
        output = torch.mean(input, 1)
    else:
        output = trident.function.mean(input, 1)

    return triton.testing.do_bench_cudagraph(
        lambda: output.backward(grad_output, retain_graph=True)
    )


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_mean_forward.run(print_data=True, show_plots=show_plots, dtype=dtype)
    else:
        bench_mean_backward.run(print_data=True, show_plots=show_plots, dtype=dtype)
