import os
import re
import threading
import time
import unittest

import numpy as np
import requests
import tritonclient.http as httpclient
from tritonclient.utils import *

_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

DEFAULT_TOTAL_PINNED_MEMORY_SIZE = 2**28
TOTAL_PINNED_MEMORY_SIZE = int(
    os.environ.get("CUSTOM_PINNED_MEMORY_POOL_SIZE", DEFAULT_TOTAL_PINNED_MEMORY_SIZE)
)
print(f"TOTAL_PINNED_MEMORY_SIZE: {TOTAL_PINNED_MEMORY_SIZE} bytes")


DEFAULT_USED_PINNED_MEMORY_SIZE = 0


def get_metrics():
    total_bytes_pattern = re.compile(r"pool_total_bytes (\d+)")
    used_bytes_pattern = re.compile(r"pool_used_bytes (\d+)")

    r = requests.get(f"http://{_tritonserver_ipaddr}:8002/metrics")
    r.raise_for_status()

    total_bytes_match = total_bytes_pattern.search(r.text)
    total_bytes_value = total_bytes_match.group(1)

    used_bytes_match = used_bytes_pattern.search(r.text)
    used_bytes_value = used_bytes_match.group(1)

    return total_bytes_value, used_bytes_value


class TestPinnedMemoryMetrics(unittest.TestCase):
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

        self.outputs = [
            httpclient.InferRequestedOutput("OUTPUT__0"),
            httpclient.InferRequestedOutput("OUTPUT__1"),
        ]

        self._assert_pinned_memory_utilization()

    def _assert_pinned_memory_utilization(self):
        total_bytes_value, used_bytes_value = get_metrics()
        self.assertEqual(int(total_bytes_value), TOTAL_PINNED_MEMORY_SIZE)
        self.assertEqual(int(used_bytes_value), DEFAULT_USED_PINNED_MEMORY_SIZE)

    def _collect_metrics(self):
        while not self.inference_completed.is_set():
            total_bytes_value, used_bytes_value = get_metrics()
            self.assertEqual(int(total_bytes_value), TOTAL_PINNED_MEMORY_SIZE)

            self.assertIn(int(used_bytes_value), [0, 64, 128, 192, 256])

    def test_pinned_memory_metrics_asynchronous_requests(self):
        with httpclient.InferenceServerClient(
            url=f"{_tritonserver_ipaddr}:8000", concurrency=10
        ) as client:
            if not client.is_model_ready(self.model_name):
                client.load_model(self.model_name)

            self._assert_pinned_memory_utilization()

            metrics_thread = threading.Thread(target=self._collect_metrics)
            metrics_thread.start()

            async_requests = []
            for _ in range(100):
                async_requests.append(
                    client.async_infer(
                        model_name=self.model_name,
                        inputs=self.inputs,
                        outputs=self.outputs,
                    )
                )

            time.sleep(1)

            for async_request in async_requests:
                async_request.get_result()

            self.inference_completed.set()

            metrics_thread.join()

        self._assert_pinned_memory_utilization()

    def test_pinned_memory_metrics_synchronous_requests(self):
        with httpclient.InferenceServerClient(
            url=f"{_tritonserver_ipaddr}:8000"
        ) as client:
            if not client.is_model_ready(self.model_name):
                client.load_model(self.model_name)

            self._assert_pinned_memory_utilization()

            metrics_thread = threading.Thread(target=self._collect_metrics)
            metrics_thread.start()

            for _ in range(100):
                response = client.infer(
                    model_name=self.model_name, inputs=self.inputs, outputs=self.outputs
                )
                response.get_response()

            time.sleep(0.1)

            self.inference_completed.set()

            metrics_thread.join()

        self._assert_pinned_memory_utilization()


if __name__ == "__main__":
    unittest.main()
