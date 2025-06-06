import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.http as client


class TrtBF16DataTypeTest(tu.TestResultCollector):
    def setUp(self):
        self.triton_client = client.InferenceServerClient(
            "localhost:8000", verbose=True
        )

    def _infer_helper(self, model_name, shape):
        inputs = []
        outputs = []
        inputs.append(client.InferInput("INPUT0", shape, "BF16"))
        inputs.append(client.InferInput("INPUT1", shape, "BF16"))

        input0_data = np.ones(shape=shape).astype(np.float32)
        input1_data = np.ones(shape=shape).astype(np.float32)

        inputs[0].set_data_from_numpy(input0_data, binary_data=True)
        inputs[1].set_data_from_numpy(input1_data, binary_data=True)

        outputs.append(client.InferRequestedOutput("OUTPUT0", binary_data=True))
        outputs.append(client.InferRequestedOutput("OUTPUT1", binary_data=True))

        results = self.triton_client.infer(model_name, inputs, outputs=outputs)

        output0_data = results.as_numpy("OUTPUT0")
        output1_data = results.as_numpy("OUTPUT1")

        np.testing.assert_equal(
            output0_data,
            input0_data + input1_data,
            "Result output does not match the expected output",
        )
        np.testing.assert_equal(
            output1_data,
            input0_data - input1_data,
            "Result output does not match the expected output",
        )

    def test_fixed(self):
        for bs in [1, 4, 8]:
            self._infer_helper(
                "plan_bf16_bf16_bf16",
                [bs, 16],
            )

        self._infer_helper(
            "plan_nobatch_bf16_bf16_bf16",
            [16],
        )

    def test_dynamic(self):
        for bs in [1, 4, 8]:
            self._infer_helper(
                "plan_bf16_bf16_bf16",
                [bs, 16, 16],
            )

        self._infer_helper(
            "plan_nobatch_bf16_bf16_bf16",
            [16, 16],
        )


if __name__ == "__main__":
    unittest.main()
