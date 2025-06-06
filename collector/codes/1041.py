import sys

sys.path.append("../common")

import unittest

import numpy as np
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException


class RestrictedAPITest(unittest.TestCase):
    def setUp(self):
        self.model_name_ = "simple"
        self.client_ = tritonhttpclient.InferenceServerClient("localhost:8000")

    def test_sanity(self):
        self.client_.get_inference_statistics("simple")
        self.client_.get_inference_statistics(
            "simple", headers={"infer-key": "infer-value"}
        )

    def test_model_repository(self):
        with self.assertRaisesRegex(InferenceServerException, "This API is restricted"):
            self.client_.unload_model(
                self.model_name_, headers={"infer-key": "infer-value"}
            )

        with self.assertRaisesRegex(
            InferenceServerException, "explicit model load / unload is not allowed"
        ):
            self.client_.unload_model(
                self.model_name_, headers={"admin-key": "admin-value"}
            )

    def test_metadata(self):
        with self.assertRaisesRegex(InferenceServerException, "This API is restricted"):
            self.client_.get_server_metadata()
        self.client_.get_server_metadata({"infer-key": "infer-value"})

    def test_infer(self):

        inputs = [
            tritonhttpclient.InferInput("INPUT0", [1, 16], "INT32"),
            tritonhttpclient.InferInput("INPUT1", [1, 16], "INT32"),
        ]
        inputs[0].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))
        inputs[1].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))

        with self.assertRaisesRegex(InferenceServerException, "This API is restricted"):
            _ = self.client_.infer(
                model_name=self.model_name_, inputs=inputs, headers={"test": "1"}
            )
        self.client_.infer(
            model_name=self.model_name_,
            inputs=inputs,
            headers={"infer-key": "infer-value"},
        )


if __name__ == "__main__":
    unittest.main()
