import concurrent.futures
import os
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class TestMaxQueueDelayTimeout(unittest.TestCase):
    def setUp(self):

        self._triton = grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001")

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

    def test_default_queue_policy_timeout_prompt_response(self):
        model_name = "dynamic_batch"
        with concurrent.futures.ThreadPoolExecutor() as pool:

            saturate_thread = pool.submit(
                self._triton.infer, model_name, self._get_inputs(batch_size=1)
            )
            time.sleep(2)

            callback, response = self._generate_callback_and_response_pair()
            self._triton.async_infer(
                model_name, self._get_inputs(batch_size=1), callback
            )
            time.sleep(2)

            time.sleep(2)
            self.assertTrue(response["responded"])
            self.assertEqual(response["result"], None)
            self.assertIsInstance(response["error"], InferenceServerException)
            self.assertEqual(response["error"].status(), "StatusCode.UNAVAILABLE")
            self.assertEqual(response["error"].message(), "Request timeout expired")

            saturate_thread.result()


if __name__ == "__main__":
    unittest.main()
