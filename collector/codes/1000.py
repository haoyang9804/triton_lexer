import os
import sys

sys.path.append("../../common")

import unittest

import numpy as np
import shm_util
import tritonclient.http as httpclient
from tritonclient.utils import *


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class EnsembleTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def infer(self, model_name):
        shape = [16]
        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000"
            ) as client:
                input_data_0 = np.random.random(shape).astype(np.float32)
                input_data_1 = np.random.random(shape).astype(np.float32)
                inputs = [
                    httpclient.InferInput(
                        "INPUT0",
                        input_data_0.shape,
                        np_to_triton_dtype(input_data_0.dtype),
                    ),
                    httpclient.InferInput(
                        "INPUT1",
                        input_data_1.shape,
                        np_to_triton_dtype(input_data_1.dtype),
                    ),
                ]
                inputs[0].set_data_from_numpy(input_data_0)
                inputs[1].set_data_from_numpy(input_data_1)
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("OUTPUT0")
                output1 = result.as_numpy("OUTPUT1")
                self.assertIsNotNone(output0)
                self.assertIsNotNone(output1)

                self.assertTrue(np.allclose(output0, 2 * input_data_0, atol=1e-06))
                self.assertTrue(np.allclose(output1, 2 * input_data_1, atol=1e-06))

    def test_ensemble(self):
        model_name = "ensemble"
        self.infer(model_name)

    def test_ensemble_gpu(self):
        model_name = "ensemble_gpu"
        self.infer(model_name)


if __name__ == "__main__":
    unittest.main()
