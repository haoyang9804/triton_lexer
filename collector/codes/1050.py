import sys

sys.path.append("../common")

import math
import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


class LargePayLoadTest(tu.TestResultCollector):
    def setUp(self):
        self._data_type = np.float32

        very_large_tensor_shape = (
            math.trunc(3 * (1024 * 1024 * 1024) / np.dtype(self._data_type).itemsize),
        )
        self._very_large_in0 = np.random.random(very_large_tensor_shape).astype(
            self._data_type
        )

        large_tensor_shape = (
            math.trunc(
                1.9 * (1024 * 1024 * 1024) // np.dtype(self._data_type).itemsize
            ),
        )
        self._large_in0 = np.random.random(large_tensor_shape).astype(self._data_type)

        small_tensor_shape = (1,)
        self._small_in0 = np.random.random(small_tensor_shape).astype(self._data_type)

        self._clients = (
            (httpclient, httpclient.InferenceServerClient("localhost:8000")),
            (grpcclient, grpcclient.InferenceServerClient("localhost:8001")),
        )

    def _test_helper(
        self, client, model_name, input_name="INPUT0", output_name="OUTPUT0"
    ):

        if not model_name.startswith("plan"):
            inputs = [
                client[0].InferInput(
                    input_name,
                    self._large_in0.shape,
                    np_to_triton_dtype(self._data_type),
                )
            ]
            inputs[0].set_data_from_numpy(self._large_in0)
            results = client[1].infer(model_name, inputs)

            self.assertTrue(
                np.array_equal(self._large_in0, results.as_numpy(output_name)),
                "output is different from input",
            )

        if client[0] == httpclient:

            inputs = [
                client[0].InferInput(
                    input_name,
                    self._very_large_in0.shape,
                    np_to_triton_dtype(self._data_type),
                )
            ]
            inputs[0].set_data_from_numpy(self._very_large_in0)
            with self.assertRaises(InferenceServerException):
                results = client[1].infer(model_name, inputs)

        inputs = [
            client[0].InferInput(
                input_name, self._small_in0.shape, np_to_triton_dtype(self._data_type)
            )
        ]
        inputs[0].set_data_from_numpy(self._small_in0)
        results = client[1].infer(model_name, inputs)
        self.assertTrue(
            np.array_equal(self._small_in0, results.as_numpy(output_name)),
            "output is different from input",
        )

    def test_onnx(self):

        for client in self._clients:
            model_name = tu.get_zero_model_name("onnx_nobatch", 1, self._data_type)
            self._test_helper(client, model_name)

    def test_python(self):

        for client in self._clients:
            model_name = tu.get_zero_model_name("python_nobatch", 1, self._data_type)
            self._test_helper(client, model_name)

    def test_plan(self):

        for client in self._clients:
            model_name = tu.get_zero_model_name("plan_nobatch", 1, self._data_type)
            self._test_helper(client, model_name)

    def test_libtorch(self):

        for client in self._clients:
            model_name = tu.get_zero_model_name("libtorch_nobatch", 1, self._data_type)
            self._test_helper(client, model_name, "INPUT__0", "OUTPUT__0")

    def test_custom(self):

        for client in self._clients:
            model_name = tu.get_zero_model_name("custom", 1, self._data_type)
            self._test_helper(client, model_name)


if __name__ == "__main__":
    unittest.main()
