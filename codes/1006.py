import os
import sys

sys.path.append("../../common")
import unittest

import numpy as np
import shm_util
import tritonclient.http as httpclient
from tritonclient.utils import *


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class LogTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def test_log_output(self):
        model_name = "identity_fp32_logging"
        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000"
            ) as client:
                input_data = np.array([[1.0]], dtype=np.float32)
                inputs = [
                    httpclient.InferInput(
                        "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("OUTPUT0")
                self.assertIsNotNone(output0)
                self.assertTrue(np.all(output0 == input_data))


if __name__ == "__main__":
    unittest.main()
