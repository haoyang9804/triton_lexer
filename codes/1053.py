import sys

sys.path.append("../common")

import unittest
from builtins import range

import numpy as np
import test_util as tu
import tritonhttpclient as httpclient

FLAGS = None


class SharedWeightsTest(tu.TestResultCollector):
    def _full_exact(self, model_name, request_concurrency, shape):

        client = httpclient.InferenceServerClient(
            "localhost:8000", concurrency=request_concurrency
        )
        input_datas = []
        requests = []
        for i in range(request_concurrency):
            input_data = (16384 * np.random.randn(*shape)).astype(np.float32)
            input_datas.append(input_data)
            inputs = [httpclient.InferInput("INPUT__0", input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_data)
            requests.append(client.async_infer(model_name, inputs))

        for i in range(request_concurrency):

            results = requests[i].get_result()

            output_data = results.as_numpy("OUTPUT__0")
            self.assertIsNotNone(output_data, "error: expected 'OUTPUT__0' to be found")
            np.testing.assert_allclose(output_data, input_datas[i])

    def test_pytorch_identity_model(self):
        model_name = "libtorch_nobatch_zero_1_float32"
        self._full_exact(model_name, 128, [8])


if __name__ == "__main__":
    unittest.main()
