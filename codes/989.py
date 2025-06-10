import os
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient


class ConcurrencyTest(unittest.TestCase):
    def setUp(self):

        self._triton = grpcclient.InferenceServerClient("localhost:8001")

    def _generate_streaming_callback_and_response_pair(self):
        response = []

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    def _concurrent_execute_requests(self, model_name, batch_size, number_of_requests):
        delay_secs = 4
        shape = [batch_size, 1]
        inputs = [grpcclient.InferInput("WAIT_SECONDS", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.full(shape, delay_secs, dtype=np.float32))

        callback, response = self._generate_streaming_callback_and_response_pair()
        self._triton.start_stream(callback)
        for i in range(number_of_requests):
            self._triton.async_stream_infer(model_name, inputs)

        wait_secs = 2 + delay_secs + 2
        time.sleep(wait_secs)

        sequential_min_delay = wait_secs * batch_size * number_of_requests
        self.assertLessEqual(wait_secs, sequential_min_delay)

        self.assertEqual(len(response), number_of_requests)
        for res in response:
            self.assertEqual(res["result"].as_numpy("DUMMY_OUT").shape[0], batch_size)
            self.assertIsNone(res["error"])

        self._triton.stop_stream()

    def test_concurrent_execute_single_request(self):
        self._concurrent_execute_requests(
            model_name="async_execute_decouple", batch_size=4, number_of_requests=1
        )

    def test_concurrent_execute_multi_request(self):
        self._concurrent_execute_requests(
            model_name="async_execute_decouple", batch_size=1, number_of_requests=4
        )

    def test_concurrent_execute_single_request_bls(self):
        self._concurrent_execute_requests(
            model_name="async_execute_decouple_bls", batch_size=4, number_of_requests=1
        )

    def test_concurrent_execute_multi_request_bls(self):
        self._concurrent_execute_requests(
            model_name="async_execute_decouple_bls", batch_size=1, number_of_requests=4
        )

    def test_concurrent_execute_different_duration(self):
        model_name = "async_execute_decouple"
        callback, response = self._generate_streaming_callback_and_response_pair()
        self._triton.start_stream(callback)

        shape = [1, 1]
        for delay_secs in [10, 2]:
            inputs = [grpcclient.InferInput("WAIT_SECONDS", shape, "FP32")]
            inputs[0].set_data_from_numpy(np.full(shape, delay_secs, dtype=np.float32))
            self._triton.async_stream_infer(model_name, inputs)
            time.sleep(2)
            shape[0] += 1

        time.sleep(4)

        self.assertEqual(len(response), 1)
        self.assertEqual(response[0]["result"].as_numpy("DUMMY_OUT").shape[0], 2)
        self.assertIsNone(response[0]["error"])

        time.sleep(6)

        self.assertEqual(len(response), 2)
        self.assertEqual(response[1]["result"].as_numpy("DUMMY_OUT").shape[0], 1)
        self.assertIsNone(response[1]["error"])

        self._triton.stop_stream()

    def test_model_raise_exception(self):
        model_name = "async_execute_decouple"
        delay_secs = -1
        shape = [1, 1]
        inputs = [grpcclient.InferInput("WAIT_SECONDS", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.full(shape, delay_secs, dtype=np.float32))

        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("ValueError: wait_secs cannot be negative", server_log)

        callback, response = self._generate_streaming_callback_and_response_pair()
        self._triton.start_stream(callback)
        self._triton.async_stream_infer(model_name, inputs)
        time.sleep(2)
        self._triton.stop_stream()

        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertIn("ValueError: wait_secs cannot be negative", server_log)


if __name__ == "__main__":
    unittest.main()
