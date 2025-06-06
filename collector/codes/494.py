import torch
import triton
import util

import trident


@util.report(
    "argmax forward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 32}
)
def bench_mean_forward(y_size, x_size, dtype, backend):
    input = torch.randn(y_size, x_size, device="cuda", dtype=dtype)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.argmax(input, 1))
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.argmax(input, 1)
        )


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_mean_forward.run(print_data=True, show_plots=show_plots, dtype=dtype)
    else:
        print("argmax backward isn't supported.")
