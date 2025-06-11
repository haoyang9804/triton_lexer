import triton
import triton.language as tl


@triton.jit
def combine_welford(m2_a, mean_a, count_a, m2_b, mean_b, count_b):
    count = count_a + count_b
    return (
        m2_a + m2_b + tl.math.pow(mean_b - mean_a, 2.0) * count_a * count_b / count,
        (mean_a * count_a + mean_b * count_b) / count,
        count,
    )


@triton.jit
def combine_softmax(
    max_a: tl.tensor, sum_a: tl.tensor, max_b: tl.tensor, sum_b: tl.tensor
):
    max = tl.math.max(max_a, max_b)
    sum = sum_a * tl.math.fast_expf(max_a - max) + sum_b * tl.math.fast_expf(
        max_b - max
    )
    return max, sum
