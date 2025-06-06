import sys

sys.path.append("../common")

import unittest

import infer_util as iu
import numpy as np
import test_util as tu
import tritonhttpclient
from tritonclientutils import InferenceServerException


class TrtDynamicShapeTest(tu.TestResultCollector):
    def setUp(self):
        self.dtype_ = np.float32
        self.model_name_ = "plan"

    def test_load_specific_optimization_profile(self):

        tensor_shape = (1,)
        try:
            iu.infer_exact(
                self,
                self.model_name_,
                (1,) + tensor_shape,
                1,
                self.dtype_,
                self.dtype_,
                self.dtype_,
            )
        except InferenceServerException as ex:
            self.assertTrue(
                "model expected the shape of dimension 0 to be between 6 and 8 but received 1"
                in ex.message()
            )

        try:
            iu.infer_exact(
                self,
                self.model_name_,
                (8,) + tensor_shape,
                8,
                self.dtype_,
                self.dtype_,
                self.dtype_,
            )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_load_default_optimization_profile(self):

        tensor_shape = (33,)

        try:
            iu.infer_exact(
                self,
                self.model_name_,
                (8,) + tensor_shape,
                8,
                self.dtype_,
                self.dtype_,
                self.dtype_,
            )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        over_tensor_shape = (34,)
        try:
            iu.infer_exact(
                self,
                self.model_name_,
                (8,) + over_tensor_shape,
                8,
                self.dtype_,
                self.dtype_,
                self.dtype_,
            )
        except InferenceServerException as ex:
            self.assertTrue(
                "model expected the shape of dimension 1 to be between 1 and 33 but received 34"
                in ex.message()
            )

    def test_select_optimization_profile(self):

        batch_size = 4
        tensor_shape = (16,)
        try:
            iu.infer_exact(
                self,
                self.model_name_,
                (batch_size,) + tensor_shape,
                batch_size,
                self.dtype_,
                self.dtype_,
                self.dtype_,
            )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_load_wrong_optimization_profile(self):
        client = tritonhttpclient.InferenceServerClient("localhost:8000")
        model_name = tu.get_model_name(
            self.model_name_, self.dtype_, self.dtype_, self.dtype_
        )
        model_status = client.is_model_ready(model_name, "1")
        self.assertFalse(model_status, "expected model to be not ready")


if __name__ == "__main__":
    unittest.main()
