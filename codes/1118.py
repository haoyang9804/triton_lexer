import sys

sys.path.append("../common")

import os
import unittest

import numpy as np
import test_util as tu
import tritonclient.http as httpclient


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


def hardmax_reference(arr, axis=0):
    one_hot = np.zeros(arr.shape, dtype=arr.dtype)
    argmax = np.expand_dims(np.argmax(arr, axis), axis)
    np.put_along_axis(one_hot, argmax, 1, axis=axis)
    return one_hot


class PluginModelTest(tu.TestResultCollector):
    def _full_exact(self, model_name, plugin_name, shape):
        print(f"{_tritonserver_ipaddr}:8000")
        triton_client = httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000")

        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput("INPUT0", list(shape), "FP32"))

        input0_data = np.ones(shape=shape).astype(np.float32)
        inputs[0].set_data_from_numpy(input0_data, binary_data=True)

        outputs.append(httpclient.InferRequestedOutput("OUTPUT0", binary_data=True))

        results = triton_client.infer(
            model_name + "_" + plugin_name, inputs, outputs=outputs
        )

        output0_data = results.as_numpy("OUTPUT0")
        tolerance_relative = 1e-6
        tolerance_absolute = 1e-7

        if plugin_name == "CustomHardmax":
            test_output = hardmax_reference(input0_data)
            np.testing.assert_allclose(
                output0_data,
                test_output,
                rtol=tolerance_relative,
                atol=tolerance_absolute,
            )
        else:
            self.fail("Unexpected plugin: " + plugin_name)

    def test_raw_hard_max(self):
        for bs in (1, 8):
            self._full_exact(
                "plan_float32_float32_float32",
                "CustomHardmax",
                (bs, 2, 2),
            )

        self._full_exact(
            "plan_nobatch_float32_float32_float32",
            "CustomHardmax",
            (16, 1, 1),
        )


if __name__ == "__main__":
    unittest.main()
