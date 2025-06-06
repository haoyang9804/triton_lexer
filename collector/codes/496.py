import torch
import triton
import util

import trident


@util.report(
    "batch norm forward",
    ["x_size"],
    [128 * i for i in range(1, 21)],
    {"num_batches": 8, "y_size": 16},
)
def bench_batch_norm_forward(num_batches, y_size, x_size, dtype, backend):
    input = torch.randn(num_batches, y_size, x_size, device="cuda", dtype=dtype)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.batch_norm(input, None, None, training=True)
        )
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.batch_norm(input, None, None, training=True)
        )


@util.report(
    "batch norm backward",
    ["x_size"],
    [128 * i for i in range(1, 21)],
    {"num_batches": 8, "y_size": 16},
)
def bench_batch_norm_backward(num_batches, y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(
        num_batches, y_size, x_size, **factory_kwargs, requires_grad=True
    )
    weight = torch.randn(y_size, **factory_kwargs, requires_grad=True)
    bias = torch.randn(y_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.randn(num_batches, y_size, x_size, **factory_kwargs)

    if backend == "torch":
        output = torch.nn.functional.batch_norm(input, None, None, weight, bias, True)
    else:
        output = trident.function.batch_norm(input, None, None, weight, bias, True)

    return triton.testing.do_bench_cudagraph(
        lambda: output.backward(grad_output, retain_graph=True)
    )


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_batch_norm_forward.run(
            print_data=True, show_plots=show_plots, dtype=dtype
        )
    else:
        bench_batch_norm_backward.run(
            print_data=True, show_plots=show_plots, dtype=dtype
        )
