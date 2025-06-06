import sys

sys.path.append("../common")

import json
import traceback
import unittest

import numpy as np
import requests
import test_util as tu
import tritonclient.grpc as tritongrpcclient
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException


class NanInfTest(tu.TestResultCollector):
    expected_output = np.array([np.nan, np.inf, np.NINF, 1, 2, 3], dtype=np.float32)
    model_name = "nan_inf_output"

    def test_http_raw(self):
        payload = {
            "inputs": [
                {"name": "INPUT0", "datatype": "FP32", "shape": [1], "data": [1]}
            ]
        }
        response = requests.post(
            "http://localhost:8000/v2/models/nan_inf_output/infer",
            data=json.dumps(payload),
        )
        if not response.ok:
            self.assertTrue(False, "Response not OK: {}".format(response.text))

        try:
            print(response.json())
        except:
            self.assertTrue(
                False, "Response was not valid JSON:\n{}".format(response.text)
            )

    def test_http(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT0", [1], "FP32"))
        self.infer_helper(triton_client, inputs)

    def test_grpc(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT0", [1], "FP32"))
        self.infer_helper(triton_client, inputs)

    def infer_helper(self, triton_client, inputs):
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))

        try:
            results = triton_client.infer(model_name=self.model_name, inputs=inputs)
            output0_data = results.as_numpy("OUTPUT0")

            output_correct = np.array_equal(
                output0_data, self.expected_output, equal_nan=True
            )
            self.assertTrue(
                output_correct, "didn't get expected output0: {}".format(output0_data)
            )
        except InferenceServerException as ex:
            self.assertTrue(False, ex.message())
        except:
            self.assertTrue(False, traceback.format_exc())


if __name__ == "__main__":
    unittest.main()
