import torch
import triton
import util

import trident


@util.report(
    "cosine similarity forward",
    ["x_size"],
    [128 * i for i in range(1, 21)],
    {"z_size": 16, "y_size": 16},
)
def bench_cosine_similarity_forward(z_size, y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    x1 = torch.randn(z_size, y_size, x_size, **factory_kwargs)
    x2 = torch.randn(z_size, y_size, x_size, **factory_kwargs)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.cosine_similarity(x1, x2, 2)
        )
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.cosine_similarity(x1, x2, 2)
        )


@util.report(
    "cosine similarity backward",
    ["x_size"],
    [128 * i for i in range(1, 21)],
    {"z_size": 16, "y_size": 16},
)
def bench_cosine_similarity_backward(z_size, y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    x1 = torch.randn(z_size, y_size, x_size, **factory_kwargs, requires_grad=True)
    x2 = torch.randn(z_size, y_size, x_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.empty(z_size, y_size, **factory_kwargs)

    if backend == "torch":
        output = torch.nn.functional.cosine_similarity(x1, x2, 2)
    else:
        output = trident.function.cosine_similarity(x1, x2, 2)

    return triton.testing.do_bench_cudagraph(
        lambda: output.backward(grad_output, retain_graph=True)
    )


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_cosine_similarity_forward.run(
            print_data=True, show_plots=show_plots, dtype=dtype
        )
    else:
        bench_cosine_similarity_backward.run(
            print_data=True, show_plots=show_plots, dtype=dtype
        )
