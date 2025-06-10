import os
import sys

sys.path.append("../common")

import math
import time
import unittest
from functools import partial

import numpy as np
import requests
import test_util as tu
import tritonclient.http
from tritonclient.utils import triton_to_np_dtype

_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

QUEUE_METRIC_TEMPLATE = (
    'nv_inference_pending_request_count{{model="{model_name}",version="1"}}'
)
INFER_METRIC_TEMPLATE = 'nv_inference_count{{model="{model_name}",version="1"}}'
EXEC_METRIC_TEMPLATE = 'nv_inference_exec_count{{model="{model_name}",version="1"}}'


class MetricsPendingRequestCountTest(tu.TestResultCollector):
    def setUp(self):
        self.metrics = None
        self.metrics_url = f"http://{_tritonserver_ipaddr}:8002/metrics"
        self.server_url = f"{_tritonserver_ipaddr}:8000"

        self.max_batch_size = 4
        self.delay_ms = 2000
        self.delay_sec = self.delay_ms // 1000

        dtype = "FP32"
        shape = (1, 1)
        input_np = np.ones(shape, dtype=triton_to_np_dtype(dtype))
        self.inputs = [
            tritonclient.http.InferInput("INPUT0", shape, dtype).set_data_from_numpy(
                input_np
            )
        ]
        self.ensemble_inputs = [
            tritonclient.http.InferInput(
                "ENSEMBLE_INPUT0", shape, dtype
            ).set_data_from_numpy(input_np)
        ]

        self.num_requests = 10
        self.concurrency = 10

        self.assertGreaterEqual(self.concurrency, self.num_requests)
        self.client = tritonclient.http.InferenceServerClient(
            url=self.server_url, concurrency=self.concurrency
        )

        self.max_queue_size = 0

    def _validate_model_config(self, model_name, max_queue_size=0):
        config = self.client.get_model_config(model_name)
        print(config)
        params = config.get("parameters", {})
        delay_ms = int(params.get("execute_delay_ms", {}).get("string_value"))
        max_batch_size = config.get("max_batch_size")
        self.assertEqual(delay_ms, self.delay_ms)
        self.assertEqual(max_batch_size, self.max_batch_size)

        dynamic_batching = config.get("dynamic_batching", {})
        default_queue_policy = dynamic_batching.get("default_queue_policy", {})
        self.max_queue_size = default_queue_policy.get("max_queue_size", 0)

        self.assertEqual(self.max_queue_size, max_queue_size)

        return config

    def _get_metrics(self):
        r = requests.get(self.metrics_url)
        r.raise_for_status()
        return r.text

    def _get_metric_line(self, metric, metrics):
        for line in metrics.splitlines():
            if metric in line:
                return line
        return None

    def _get_metric_value(self, metric):
        metrics = self._get_metrics()
        self.assertIn(metric, metrics)
        line = self._get_metric_line(metric, metrics)
        print(line)
        if not line:
            return None
        value = line.split()[1]
        return float(value)

    def _assert_metric_equals(self, metric, expected_value):
        value = self._get_metric_value(metric)
        self.assertEqual(value, expected_value)

    def _assert_metric_greater_than(self, metric, gt_value):
        value = self._get_metric_value(metric)
        self.assertGreater(value, gt_value)

    def _send_async_requests(self, model_name, inputs, futures):
        for _ in range(self.num_requests):
            futures.append(self.client.async_infer(model_name, inputs))

    def _send_async_requests_sequence(self, num_seq_slots, model_name, inputs, futures):
        started_seqs = {}
        num_sent = 0
        while num_sent < self.num_requests:

            seq_id = (num_sent % num_seq_slots) + 1

            start = True if seq_id not in started_seqs else False
            started_seqs[seq_id] = True
            futures.append(
                self.client.async_infer(
                    model_name,
                    inputs,
                    request_id=str(num_sent),
                    sequence_id=seq_id,
                    sequence_start=start,
                )
            )
            num_sent += 1

    def _test_helper(
        self, model_name, batch_size, send_requests_func, max_queue_size=0
    ):
        self._validate_model_config(model_name, max_queue_size=max_queue_size)

        queue_size = QUEUE_METRIC_TEMPLATE.format(model_name=model_name)
        infer_count = INFER_METRIC_TEMPLATE.format(model_name=model_name)
        exec_count = EXEC_METRIC_TEMPLATE.format(model_name=model_name)

        self._assert_metric_equals(queue_size, 0)

        futures = []
        send_requests_func(model_name, self.inputs, futures)

        time.sleep(1)

        if max_queue_size != 0:
            self._assert_metric_equals(queue_size, max_queue_size)
            starting_queue_size = max_queue_size
        else:
            starting_queue_size = self.num_requests - batch_size

        for expected_queue_size in range(starting_queue_size, 0, -1 * batch_size):
            self._assert_metric_equals(queue_size, expected_queue_size)
            time.sleep(self.delay_sec)

        self._assert_metric_equals(queue_size, 0)

        time.sleep(self.delay_sec)

        expected_infer_count = starting_queue_size + batch_size
        self._assert_metric_equals(infer_count, expected_infer_count)
        expected_exec_count = math.ceil(expected_infer_count / batch_size)
        self._assert_metric_equals(exec_count, expected_exec_count)

        failed_count = 0
        for future in futures:
            try:
                future.get_result()
            except Exception as e:
                failed_count = failed_count + 1

        self.assertEqual(
            failed_count, self.num_requests - batch_size - starting_queue_size
        )

    def test_default_scheduler(self):
        model_name = "default"

        batch_size = 1
        self._test_helper(model_name, batch_size, self._send_async_requests)

    def test_dynamic_batch_scheduler(self):
        model_name = "dynamic"

        batch_size = self.max_batch_size
        self._test_helper(model_name, batch_size, self._send_async_requests)

    def test_fail_max_queue_size(self):
        model_name = "max_queue_size"

        batch_size = self.max_batch_size
        self._test_helper(
            model_name, batch_size, self._send_async_requests, max_queue_size=4
        )

    def test_sequence_batch_scheduler_direct(self):
        model_name = "sequence_direct"

        batch_size = self.max_batch_size
        num_seq_slots = batch_size
        send_requests_func = partial(self._send_async_requests_sequence, num_seq_slots)
        self._test_helper(model_name, batch_size, send_requests_func)

    def test_sequence_batch_scheduler_oldest(self):
        model_name = "sequence_oldest"

        batch_size = self.max_batch_size
        num_seq_slots = batch_size
        send_requests_func = partial(self._send_async_requests_sequence, num_seq_slots)
        self._test_helper(model_name, batch_size, send_requests_func)

    def test_ensemble_scheduler(self):
        ensemble_model_name = "ensemble"
        composing_model_names = ["dynamic_composing", "default_composing"]
        ensemble_queue_size = QUEUE_METRIC_TEMPLATE.format(
            model_name=ensemble_model_name
        )
        composing_queue_sizes = [
            QUEUE_METRIC_TEMPLATE.format(model_name=name)
            for name in composing_model_names
        ]
        ensemble_infer_count = INFER_METRIC_TEMPLATE.format(
            model_name=ensemble_model_name
        )
        composing_infer_counts = [
            INFER_METRIC_TEMPLATE.format(model_name=name)
            for name in composing_model_names
        ]

        self._assert_metric_equals(ensemble_queue_size, 0)
        for queue_size in composing_queue_sizes:
            self._assert_metric_equals(queue_size, 0)

        futures = []
        self._send_async_requests(ensemble_model_name, self.ensemble_inputs, futures)

        time.sleep(1)

        self._assert_metric_equals(ensemble_queue_size, 0)

        for queue_size in composing_queue_sizes:
            self._assert_metric_greater_than(queue_size, 0)

        for future in futures:
            future.get_result()

        self._assert_metric_equals(ensemble_queue_size, 0)
        for queue_size in composing_queue_sizes:
            self._assert_metric_equals(queue_size, 0)

        self._assert_metric_equals(ensemble_infer_count, self.num_requests)
        for infer_count in composing_infer_counts:
            self._assert_metric_equals(infer_count, self.num_requests)


if __name__ == "__main__":
    unittest.main()
