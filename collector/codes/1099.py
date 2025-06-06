import sys

sys.path.append("../common")

import os
import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


class ScalarIOTest(tu.TestResultCollector):
    def setUp(self):
        self._client = grpcclient.InferenceServerClient(url="localhost:8001")
        self._backends = os.environ.get("BACKENDS", "onnx").split(",")

    def _send_request_and_verify_result(self, input, model_name):
        inputs = []
        inputs.append(
            grpcclient.InferInput("INPUT", input.shape, np_to_triton_dtype(input.dtype))
        )
        inputs[-1].set_data_from_numpy(input)
        result = self._client.infer(inputs=inputs, model_name=model_name)
        output = result.as_numpy("OUTPUT")
        np.testing.assert_allclose(input, output)

    def test_scalar_io(self):
        for backend in self._backends:
            model_name = f"{backend}_scalar_1dim"
            self._send_request_and_verify_result(
                np.asarray([1], dtype=np.float32), model_name
            )

            model_name = f"{backend}_scalar_2dim"
            self._send_request_and_verify_result(
                np.asarray([[1]], dtype=np.float32), model_name
            )


if __name__ == "__main__":
    unittest.main()
