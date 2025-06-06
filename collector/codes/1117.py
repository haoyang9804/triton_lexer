import unittest

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class TestTrtErrorPropagation(unittest.TestCase):
    def setUp(self):

        self.__triton = grpcclient.InferenceServerClient("localhost:8001", verbose=True)

    def test_invalid_trt_model(self):
        with self.assertRaises(InferenceServerException) as cm:
            self.__triton.load_model("invalid_plan_file")
        err_msg = str(cm.exception)

        expected_msg_parts = [
            "load failed for model",
            "version 1 is at UNAVAILABLE state: ",
            "Internal: unable to create TensorRT engine: ",
            "Error Code ",
            "Internal Error ",
        ]
        for expected_msg_part in expected_msg_parts:
            self.assertIn(
                expected_msg_part,
                err_msg,
                "Cannot find an expected part of error message",
            )
            _, err_msg = err_msg.split(expected_msg_part)

    def test_invalid_trt_model_autocomplete(self):
        with self.assertRaises(InferenceServerException) as cm:
            self.__triton.load_model("invalid_plan_file")
        err_msg = str(cm.exception)
        self.assertIn(
            "Internal: unable to load plan file to auto complete config",
            err_msg,
            "Caught an unexpected exception",
        )


if __name__ == "__main__":
    unittest.main()
