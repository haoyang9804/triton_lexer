import os
import re
import threading
import time
import unittest
from collections import defaultdict

import numpy as np
import requests
import tritonclient.http as httpclient

_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")
CPU_UTILIZATION = "nv_cpu_utilization"
CPU_USED_MEMORY = "nv_cpu_memory_used_bytes"
CPU_TOTAL_MEMORY = "nv_cpu_memory_total_bytes"


def get_metrics():
    utilization_pattern = re.compile(rf"{CPU_UTILIZATION} (\d+\.?\d*)")
    used_bytes_pattern = re.compile(rf"{CPU_USED_MEMORY} (\d+)")
    total_bytes_pattern = re.compile(rf"{CPU_TOTAL_MEMORY} (\d+)")

    r = requests.get(f"http://{_tritonserver_ipaddr}:8002/metrics")
    r.raise_for_status()

    utilization_match = utilization_pattern.search(r.text)
    utilization_value = float(utilization_match.group(1))

    used_bytes_match = used_bytes_pattern.search(r.text)
    used_bytes_value = int(used_bytes_match.group(1))

    total_bytes_match = total_bytes_pattern.search(r.text)
    total_bytes_value = int(total_bytes_match.group(1))

    return utilization_value, used_bytes_value, total_bytes_value


class TestCpuMetrics(unittest.TestCase):
    def setUp(self):
        self.inference_completed = threading.Event()

        shape = [1, 16]
        self.model_name = "libtorch_float32_float32_float32"
        input0_data = np.random.rand(*shape).astype(np.float32)
        input1_data = np.random.rand(*shape).astype(np.float32)

        self.inputs = [
            httpclient.InferInput(
                "INPUT0", input0_data.shape, "FP32"
            ).set_data_from_numpy(input0_data),
            httpclient.InferInput(
                "INPUT1", input1_data.shape, "FP32"
            ).set_data_from_numpy(input1_data),
        ]

    def _validate_metric_variance(self, observed_metrics: dict):
        dupe_value_tolerance = 5
        for metric in [CPU_UTILIZATION, CPU_USED_MEMORY]:
            observed_values = observed_metrics[metric]
            observed_count = len(observed_values)
            print(
                f"Observed {metric} count: {observed_count}, values: {observed_values}"
            )

            self.assertGreater(
                observed_count,
                dupe_value_tolerance,
                f"Found too many sequential duplicate values for {metric}. Double check the server-side --metrics-interval and observation interval in this test, or consider tuning the duplicate tolerance.",
            )

            sequential_dupes = 0
            max_sequential_dupes = 0
            prev_value = observed_values[0]
            for value in observed_values[1:]:
                if value == prev_value:
                    sequential_dupes += 1
                else:

                    sequential_dupes = 0

                if sequential_dupes > max_sequential_dupes:
                    max_sequential_dupes = sequential_dupes

                self.assertLess(sequential_dupes, dupe_value_tolerance)
                prev_value = value

            print(
                f"Max sequential duplicate values found for {metric}: {max_sequential_dupes}"
            )

    def _collect_metrics(self, observed_metrics, interval_secs=1):

        time.sleep(1)

        while not self.inference_completed.is_set():
            util_value, used_memory_value, _ = get_metrics()
            observed_metrics[CPU_UTILIZATION].append(util_value)
            observed_metrics[CPU_USED_MEMORY].append(used_memory_value)
            time.sleep(interval_secs)

    def test_cpu_metrics_during_inference(self):
        with httpclient.InferenceServerClient(
            url=f"{_tritonserver_ipaddr}:8000", concurrency=10
        ) as client:

            observed_metrics = defaultdict(list)
            metrics_thread = threading.Thread(
                target=self._collect_metrics, args=(observed_metrics,)
            )
            metrics_thread.start()

            async_requests = []
            for _ in range(2000):
                async_requests.append(
                    client.async_infer(
                        model_name=self.model_name,
                        inputs=self.inputs,
                    )
                )

            for async_request in async_requests:
                async_request.get_result()

            self.inference_completed.set()

            metrics_thread.join()

        self._validate_metric_variance(observed_metrics)

    def test_cpu_metrics_ranges(self):

        utilization, used_memory, total_memory = get_metrics()
        self.assertTrue(0 <= utilization <= 1.0)
        self.assertTrue(0 <= used_memory <= total_memory)

        self.assertGreater(total_memory, 0)


if __name__ == "__main__":
    unittest.main()
