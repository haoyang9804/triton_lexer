import torch
import triton
import util

import trident


@util.report(
    "layer norm forward",
    ["x_size"],
    [128 * i for i in range(1, 21)],
    {"num_batches": 512, "y_size": 2048},
)
def bench_layer_norm_forward(num_batches, y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(num_batches, y_size, x_size, **factory_kwargs)
    normalized_shape = (input.shape[-1],)
    weight = torch.randn(x_size, **factory_kwargs)
    bias = torch.randn(x_size, **factory_kwargs)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.layer_norm(
                input, normalized_shape, weight, bias
            )
        )
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.layer_norm(input, normalized_shape, weight, bias)
        )


@util.report(
    "layer norm backward",
    ["x_size"],
    [128 * i for i in range(1, 21)],
    {"num_batches": 512, "y_size": 2048},
)
def bench_layer_norm_backward(num_batches, y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype, "requires_grad": True}
    input = torch.randn(num_batches, y_size, x_size, **factory_kwargs)
    weight = torch.randn(x_size, **factory_kwargs)
    bias = torch.randn(x_size, **factory_kwargs)
    normalized_shape = (input.shape[-1],)
    grad_output = torch.randn((num_batches, y_size, x_size), device="cuda", dtype=dtype)

    if backend == "torch":
        output = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias)
    else:
        output = trident.function.layer_norm(input, normalized_shape, weight, bias)

    return triton.testing.do_bench_cudagraph(
        lambda: output.backward(grad_output, retain_graph=True)
    )


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_layer_norm_forward.run(
            print_data=True, show_plots=show_plots, dtype=dtype
        )
    else:
        bench_layer_norm_backward.run(
            print_data=True, show_plots=show_plots, dtype=dtype
        )
