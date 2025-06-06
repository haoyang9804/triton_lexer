import os
import sys

sys.path.append("../../common")

import unittest

import numpy as np
import shm_util
import tritonclient.http as httpclient
from tritonclient.utils import *


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class RestartTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def _infer_helper(self, model_name, shape, data_type):
        with httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000") as client:
            input_data_0 = np.array(np.random.randn(*shape), dtype=data_type)
            inputs = [
                httpclient.InferInput(
                    "INPUT0", shape, np_to_triton_dtype(input_data_0.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data_0)
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("OUTPUT0")
            self.assertTrue(np.all(input_data_0 == output0))

    def test_restart(self):
        shape = [1, 16]
        model_name = "restart"
        dtype = np.float32

        with self.assertRaises(InferenceServerException):

            self._infer_helper(model_name, shape, dtype)

        with self._shm_leak_detector.Probe() as shm_probe:
            self._infer_helper(model_name, shape, dtype)

    def test_infer(self):
        shape = [1, 16]
        model_name = "restart"
        dtype = np.float32
        with self._shm_leak_detector.Probe() as shm_probe:
            self._infer_helper(model_name, shape, dtype)


if __name__ == "__main__":
    unittest.main()
