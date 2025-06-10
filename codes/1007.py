import os
import sys

sys.path.append("../../common")

import unittest

import numpy as np
import shm_util
import tritonclient.http as httpclient
from tritonclient.utils import *


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class ExplicitModelTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def send_identity_request(self, client, model_name):
        inputs = []
        inputs.append(httpclient.InferInput("INPUT0", [1, 16], "FP32"))
        input0_data = np.arange(start=0, stop=16, dtype=np.float32)
        input0_data = np.expand_dims(input0_data, axis=0)
        inputs[0].set_data_from_numpy(input0_data)

        with self._shm_leak_detector.Probe() as shm_probe:
            result = client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=[httpclient.InferRequestedOutput("OUTPUT0")],
            )
        output_numpy = result.as_numpy("OUTPUT0")
        self.assertTrue(np.all(input0_data == output_numpy))

    def test_model_reload(self):
        model_name = "identity_fp32"
        ensemble_model_name = "simple_" + "identity_fp32"
        with httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000") as client:
            for _ in range(5):
                self.assertFalse(client.is_model_ready(model_name))

                client.load_model(model_name)
                client.load_model(ensemble_model_name)
                self.assertTrue(client.is_model_ready(model_name))
                self.assertTrue(client.is_model_ready(ensemble_model_name))
                self.send_identity_request(client, model_name)
                self.send_identity_request(client, ensemble_model_name)
                client.unload_model(ensemble_model_name)
                client.unload_model(model_name)
                self.assertFalse(client.is_model_ready(model_name))
                self.assertFalse(client.is_model_ready(ensemble_model_name))


if __name__ == "__main__":
    unittest.main()
