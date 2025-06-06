import json
import os
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class TestBlsParameters(unittest.TestCase):
    def test_bls_parameters(self):
        model_name = "bls_parameters"
        shape = [1]
        num_params = 3

        expected_params = {}
        for i in range(1, num_params + 1):
            expected_params["bool_" + str(i)] = bool(i)
            expected_params["int_" + str(i)] = i
            expected_params["str_" + str(i)] = str(i)

        with grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001") as client:
            input_data = np.array([num_params], dtype=np.ubyte)
            inputs = [
                grpcclient.InferInput(
                    "NUMBER_PARAMETERS", shape, np_to_triton_dtype(input_data.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [grpcclient.InferRequestedOutput("PARAMETERS_AGGREGATED")]
            result = client.infer(model_name, inputs, outputs=outputs)
            params_json = str(
                result.as_numpy("PARAMETERS_AGGREGATED")[0], encoding="utf-8"
            )

        params = json.loads(params_json)
        self.assertEqual(params, expected_params)


if __name__ == "__main__":
    unittest.main()
