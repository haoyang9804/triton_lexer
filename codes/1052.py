import sys

sys.path.append("../common")

import unittest
from builtins import range

import numpy as np
import test_util as tu
import tritonclient.http as httpclient


class IONamingConvention(tu.TestResultCollector):
    def _infer_helper(self, model_name, io_names, reversed_order=False):
        triton_client = httpclient.InferenceServerClient(
            "localhost:8000", verbose=False
        )

        input0_data = np.arange(start=0, stop=16, dtype=np.float32)
        input0_data = np.expand_dims(input0_data, axis=0)
        input1_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.float32)

        inputs = []
        output_req = []
        inputs.append(
            httpclient.InferInput(
                io_names[0] if not reversed_order else io_names[1], [1, 16], "FP32"
            )
        )
        inputs[-1].set_data_from_numpy(input0_data)
        inputs.append(
            httpclient.InferInput(
                io_names[1] if not reversed_order else io_names[0], [1, 16], "FP32"
            )
        )
        inputs[-1].set_data_from_numpy(input1_data)
        output_req.append(
            httpclient.InferRequestedOutput(io_names[2], binary_data=True)
        )
        output_req.append(
            httpclient.InferRequestedOutput(io_names[3], binary_data=True)
        )

        results = triton_client.infer(model_name, inputs, outputs=output_req)

        output0_data = results.as_numpy(
            io_names[2] if not reversed_order else io_names[3]
        )
        output1_data = results.as_numpy(
            io_names[3] if not reversed_order else io_names[2]
        )
        for i in range(16):
            self.assertEqual(input0_data[0][i] - input1_data[0][i], output0_data[0][i])
            self.assertEqual(input0_data[0][i] + input1_data[0][i], output1_data[0][i])

    def test_io_index(self):
        io_names = ["INPUT__0", "INPUT__1", "OUTPUT__0", "OUTPUT__1"]
        self._infer_helper("libtorch_io_index", io_names)

    def test_output_index(self):
        io_names = ["INPUT0", "INPUT1", "OUTPUT__0", "OUTPUT__1"]
        self._infer_helper("libtorch_output_index", io_names)

    def test_no_output_index(self):
        io_names = ["INPUT0", "INPUT1", "OUTPUT0", "OUTPUT1"]
        self._infer_helper("libtorch_no_output_index", io_names)

    def test_no_arguments_no_output_index(self):
        io_names = ["INPUTA", "INPUTB", "OUTPUTA", "OUTPUTB"]
        self._infer_helper("libtorch_no_arguments_output_index", io_names)

    def test_mix_index(self):
        io_names = ["INPUTA", "INPUT__1", "OUTPUTA", "OUTPUT__1"]
        self._infer_helper("libtorch_mix_index", io_names)

    def test_mix_arguments(self):
        io_names = ["INPUT0", "INPUTB", "OUTPUTA", "OUTPUT__1"]
        self._infer_helper("libtorch_mix_arguments", io_names)

    def test_mix_arguments_index(self):
        io_names = ["INPUT0", "INPUT__1", "OUTPUT0", "OUTPUT__1"]
        self._infer_helper("libtorch_mix_arguments_index", io_names)

    def test_unordered_index(self):
        io_names = ["INPUT1", "INPUT0", "OUT__1", "OUT__0"]
        self._infer_helper("libtorch_unordered_index", io_names, reversed_order=True)


if __name__ == "__main__":
    unittest.main()
