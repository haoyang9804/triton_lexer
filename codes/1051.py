import os
import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.http as httpclient


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class InferTest(tu.TestResultCollector):
    def test_infer(self):
        try:
            triton_client = httpclient.InferenceServerClient(
                url=f"{_tritonserver_ipaddr}:8000"
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        model_name = os.environ["MODEL_NAME"]

        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput("INPUT0", [1, 16], "FP32"))
        inputs.append(httpclient.InferInput("INPUT1", [1, 16], "FP32"))

        input0_data = np.arange(start=0, stop=16, dtype=np.float32)
        input0_data = np.expand_dims(input0_data, axis=0)
        input1_data = np.arange(start=32, stop=48, dtype=np.float32)
        input1_data = np.expand_dims(input1_data, axis=0)

        inputs[0].set_data_from_numpy(input0_data, binary_data=True)
        inputs[1].set_data_from_numpy(input1_data, binary_data=True)

        outputs.append(httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True))
        outputs.append(httpclient.InferRequestedOutput("OUTPUT__1", binary_data=True))

        results = triton_client.infer(model_name, inputs, outputs=outputs)

        output0_data = results.as_numpy("OUTPUT__0")
        output1_data = results.as_numpy("OUTPUT__1")

        expected_output_0 = input0_data + input1_data
        expected_output_1 = input0_data - input1_data

        self.assertEqual(output0_data.shape, (1, 16))
        self.assertEqual(output1_data.shape, (1, 16))

        self.assertTrue(np.all(expected_output_0 == output0_data))
        self.assertTrue(np.all(expected_output_1 == output1_data))


if __name__ == "__main__":
    unittest.main()
