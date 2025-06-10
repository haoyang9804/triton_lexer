import sys

sys.path.append("../common")

import unittest

import infer_util as iu
import numpy as np
import test_util as tu
from tritonclient.utils import *


class TrtCudaGraphTest(tu.TestResultCollector):
    MODELNAME = "plan"

    def setUp(self):
        self.dtype_ = np.float32
        self.dtype_str_ = "FP32"
        self.model_name_ = self.MODELNAME

    def _check_infer(self, tensor_shape, batch_size=1):
        try:
            if batch_size:
                full_shape = (batch_size,) + tensor_shape
            else:
                full_shape = tensor_shape
            iu.infer_exact(
                self,
                self.model_name_,
                full_shape,
                batch_size,
                self.dtype_,
                self.dtype_,
                self.dtype_,
                model_version=1,
                use_http_json_tensors=False,
                use_grpc=False,
                use_streaming=False,
            )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def _erroneous_infer(self, tensor_shape, batch_size):
        import tritonhttpclient

        item_size = batch_size
        for dim in tensor_shape:
            item_size *= dim
        full_shape = (batch_size,) + tensor_shape
        input_np = np.arange(item_size, dtype=self.dtype_).reshape(full_shape)
        expected_output0_np = input_np + input_np
        expected_output1_np = input_np - input_np

        inputs = []
        inputs.append(
            tritonhttpclient.InferInput("INPUT0", full_shape, self.dtype_str_)
        )
        inputs[-1].set_data_from_numpy(input_np)
        inputs.append(
            tritonhttpclient.InferInput("INPUT1", full_shape, self.dtype_str_)
        )
        inputs[-1].set_data_from_numpy(input_np)
        outputs = []
        outputs.append(
            tritonhttpclient.InferRequestedOutput("OUTPUT0", binary_data=True)
        )
        outputs.append(
            tritonhttpclient.InferRequestedOutput("OUTPUT1", binary_data=True)
        )

        model_name = tu.get_model_name(
            self.model_name_, self.dtype_, self.dtype_, self.dtype_
        )
        results = tritonhttpclient.InferenceServerClient(
            "localhost:8000", verbose=True
        ).infer(model_name=model_name, inputs=inputs, outputs=outputs)

        output0_np = results.as_numpy("OUTPUT0")
        output1_np = results.as_numpy("OUTPUT1")
        self.assertFalse(
            np.array_equal(output0_np, expected_output0_np),
            "expects OUTPUT0 is not correct",
        )
        self.assertFalse(
            np.array_equal(output1_np, expected_output1_np),
            "expects OUTPUT1 is not correct",
        )

    def test_fixed_shape(self):
        tensor_shape = (16,)
        self._check_infer(tensor_shape)

        self._check_infer(tensor_shape, 5)

    def test_dynamic_shape(self):
        tensor_shape = (16,)
        self._check_infer(tensor_shape)

        self._check_infer((20,))
        self._check_infer(tensor_shape, 5)

    def test_range_fixed_shape(self):
        tensor_shape = (16,)

        self._check_infer(tensor_shape, 4)
        self._check_infer(tensor_shape, 2)

        self._check_infer(tensor_shape, 1)
        self._check_infer(tensor_shape, 8)

    def test_range_dynamic_shape(self):

        self._check_infer((16,), 4)
        self._check_infer((16,), 2)

        self._erroneous_infer((10,), 3)

        self._check_infer((7,), 3)
        self._check_infer((16,), 1)
        self._check_infer((16,), 8)
        self._check_infer((30,), 4)

    def test_nobatch_fixed_shape(self):
        self._check_infer((16,), 0)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        TrtCudaGraphTest.MODELNAME = sys.argv.pop()

    unittest.main()
