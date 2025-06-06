import concurrent.futures
import re
import time
import unittest

import numpy as np
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class TestScheduler(unittest.TestCase):
    def setUp(self):

        self._triton = grpcclient.InferenceServerClient("localhost:8001")

    def _get_inputs(self, batch_size):
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        shape = [batch_size, 8]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        return inputs

    def _generate_callback_and_response_pair(self):
        response = {"responded": False, "result": None, "error": None}

        def callback(result, error):
            response["responded"] = True
            response["result"] = result
            response["error"] = error

        return callback, response

    def _assert_response_is_cancelled(self, response):
        self.assertTrue(response["responded"])
        self.assertEqual(response["result"], None)
        self.assertIsInstance(response["error"], InferenceServerException)
        self.assertEqual(response["error"].status(), "StatusCode.CANCELLED")

    def _generate_streaming_callback_and_response_pair(self):
        response = []

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    def _assert_streaming_response_is_cancelled(self, response):
        self.assertGreater(len(response), 0)
        cancelled_count = 0
        for res in response:
            result, error = res["result"], res["error"]
            if error:
                self.assertEqual(result, None)
                self.assertIsInstance(error, InferenceServerException)
                if error.status() == "StatusCode.CANCELLED":
                    cancelled_count += 1
        self.assertEqual(cancelled_count, 1)

    def _get_metrics(self):
        metrics_url = "http://localhost:8002/metrics"
        r = requests.get(metrics_url)
        r.raise_for_status()
        return r.text

    def _metrics_before_test(self, model, reason):
        pattern = rf'nv_inference_request_failure\{{model="{model}",reason="{reason}",version="1"\}} (\d+)'
        metrics = self._get_metrics()
        match = re.search(pattern, metrics)
        if match:
            return int(match.group(1))
        else:
            raise Exception(f"Failure metrics for model='{model}' not found")

    def _assert_metrics(
        self, model_name, reason, expected_count_increase, initial_count
    ):
        metrics = self._get_metrics()

        expected_metric = f'nv_inference_request_failure{{model="{model_name}",reason="{reason}",version="1"}} {expected_count_increase + initial_count}'
        self.assertIn(expected_metric, metrics)

    def test_dynamic_batch_scheduler_request_cancellation(self):
        model_name = "dynamic_batch"
        with concurrent.futures.ThreadPoolExecutor() as pool:

            saturate_thread_1 = pool.submit(
                self._triton.infer, model_name, self._get_inputs(batch_size=1)
            )
            saturate_thread_2 = pool.submit(
                self._triton.infer, model_name, self._get_inputs(batch_size=1)
            )
            time.sleep(2)

            callback, response = self._generate_callback_and_response_pair()
            queue_future = self._triton.async_infer(
                model_name, self._get_inputs(batch_size=1), callback
            )
            time.sleep(2)
            self.assertFalse(response["responded"])

            queue_future.cancel()
            time.sleep(2)
            self._assert_response_is_cancelled(response)

            saturate_thread_1.result()
            saturate_thread_2.result()

    def test_sequence_batch_scheduler_backlog_request_cancellation(self):
        model_name = "sequence_direct"
        initial_metrics_value = self._metrics_before_test(model_name, "CANCELED")
        with concurrent.futures.ThreadPoolExecutor() as pool:

            saturate_thread = pool.submit(
                self._triton.infer,
                model_name,
                self._get_inputs(batch_size=1),
                sequence_id=1,
                sequence_start=True,
            )
            time.sleep(2)

            backlog_requests = []
            for i in range(2):
                callback, response = self._generate_callback_and_response_pair()
                backlog_future = self._triton.async_infer(
                    model_name,
                    self._get_inputs(batch_size=1),
                    callback,
                    sequence_id=2,
                    sequence_start=(True if i == 0 else False),
                )
                backlog_requests.append(
                    {"future": backlog_future, "response": response}
                )
            time.sleep(2)
            self.assertFalse(backlog_requests[0]["response"]["responded"])
            self.assertFalse(backlog_requests[1]["response"]["responded"])

            backlog_requests[0]["future"].cancel()
            time.sleep(2)
            time.sleep(2)
            self._assert_response_is_cancelled(backlog_requests[0]["response"])
            self._assert_response_is_cancelled(backlog_requests[1]["response"])

            saturate_thread.result()
        expected_count_increase = 2
        self._assert_metrics(
            model_name,
            "CANCELED",
            expected_count_increase,
            initial_metrics_value,
        )

    def test_direct_sequence_batch_scheduler_request_cancellation(self):
        model_name = "sequence_direct"
        initial_metrics_value = self._metrics_before_test(model_name, "CANCELED")
        self._test_sequence_batch_scheduler_queued_request_cancellation(model_name)
        expected_count_increase = 2
        self._assert_metrics(
            model_name,
            "CANCELED",
            expected_count_increase,
            initial_metrics_value,
        )

    def test_oldest_sequence_batch_scheduler_request_cancellation(self):
        model_name = "sequence_oldest"
        self._test_sequence_batch_scheduler_queued_request_cancellation(model_name)

    def _test_sequence_batch_scheduler_queued_request_cancellation(self, model_name):
        with concurrent.futures.ThreadPoolExecutor() as pool:

            start_thread = pool.submit(
                self._triton.infer,
                model_name,
                self._get_inputs(batch_size=1),
                sequence_id=1,
                sequence_start=True,
            )
            time.sleep(2)

            queue_requests = []
            for i in range(2):
                callback, response = self._generate_callback_and_response_pair()
                queue_future = self._triton.async_infer(
                    model_name, self._get_inputs(batch_size=1), callback, sequence_id=1
                )
                queue_requests.append({"future": queue_future, "response": response})
            time.sleep(2)
            self.assertFalse(queue_requests[0]["response"]["responded"])
            self.assertFalse(queue_requests[1]["response"]["responded"])

            queue_requests[0]["future"].cancel()
            time.sleep(2)
            time.sleep(2)
            self._assert_response_is_cancelled(queue_requests[0]["response"])
            self._assert_response_is_cancelled(queue_requests[1]["response"])

            start_thread.result()

    def test_ensemble_scheduler_request_cancellation(self):
        model_name = "ensemble_model"
        callback, response = self._generate_callback_and_response_pair()
        infer_future = self._triton.async_infer(
            model_name, self._get_inputs(batch_size=1), callback
        )
        time.sleep(2)
        self.assertFalse(response["responded"])
        infer_future.cancel()
        time.sleep(2)
        self._assert_response_is_cancelled(response)

    def test_scheduler_streaming_request_cancellation(self):
        model_name = "sequence_oldest"

        callback, response = self._generate_streaming_callback_and_response_pair()
        self._triton.start_stream(callback)
        for sequence_id in [1, 2]:
            sequence_start = True
            for request_id in range(16):
                self._triton.async_stream_infer(
                    model_name,
                    self._get_inputs(batch_size=1),
                    sequence_id=sequence_id,
                    sequence_start=sequence_start,
                )
                sequence_start = False
        time.sleep(2)

        self._triton.stop_stream(cancel_requests=True)
        time.sleep(2)
        time.sleep(2)
        self._assert_streaming_response_is_cancelled(response)


if __name__ == "__main__":
    unittest.main()
