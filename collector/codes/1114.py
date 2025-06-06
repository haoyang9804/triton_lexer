import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.http as client


class TrtDataDependentShapeTest(tu.TestResultCollector):
    def setUp(self):
        self.triton_client = client.InferenceServerClient(
            "localhost:8000", verbose=True
        )

    def test_fixed(self):
        model_name = "plan_nobatch_nonzero_fixed"
        input_np = np.arange(16, dtype=np.int32).reshape((4, 4))
        expected_output_np = np.nonzero(input_np)

        inputs = []
        inputs.append(client.InferInput("INPUT", [4, 4], "INT32"))
        inputs[-1].set_data_from_numpy(input_np)

        results = self.triton_client.infer(model_name=model_name, inputs=inputs)

        output_np = results.as_numpy("OUTPUT")
        self.assertTrue(
            np.array_equal(output_np, expected_output_np),
            "OUTPUT expected: {}, got {}".format(expected_output_np, output_np),
        )

    def test_dynamic(self):
        model_name = "plan_nobatch_nonzero_dynamic"
        input_data = []
        for i in range(20 * 16):
            input_data.append(i if (i % 2) == 0 else 0)
        input_np = np.array(input_data, dtype=np.int32).reshape((20, 16))
        expected_output_np = np.nonzero(input_np)

        inputs = []
        inputs.append(client.InferInput("INPUT", [20, 16], "INT32"))
        inputs[-1].set_data_from_numpy(input_np)

        results = self.triton_client.infer(model_name=model_name, inputs=inputs)

        output_np = results.as_numpy("OUTPUT")
        self.assertTrue(
            np.array_equal(output_np, expected_output_np),
            "OUTPUT expected: {}, got {}".format(expected_output_np, output_np),
        )


if __name__ == "__main__":
    unittest.main()
