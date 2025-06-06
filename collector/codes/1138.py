import time
import unittest

import numpy as np
import triton_python_backend_utils as pb_utils


class PBBLSModelLoadingTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "onnx_int32_int32_int32"

    def tearDown(self):

        pb_utils.unload_model(self.model_name)

        print("Sleep 30 seconds to make sure model finishes unloading...")
        time.sleep(30)
        print("Done sleeping.")

    def test_load_unload_model(self):
        self.assertFalse(pb_utils.is_model_ready(model_name=self.model_name))
        pb_utils.load_model(model_name=self.model_name)
        self.assertTrue(pb_utils.is_model_ready(self.model_name))
        pb_utils.unload_model(self.model_name)
        self.assertFalse(pb_utils.is_model_ready(self.model_name))

    def test_load_with_config_override(self):
        self.assertFalse(pb_utils.is_model_ready(self.model_name))
        pb_utils.load_model(self.model_name)
        self.assertTrue(pb_utils.is_model_ready(self.model_name))

        wrong_config = '"parameters": {"config": {{"backend":"onnxruntime", "version_policy":{"specific":{"versions":[2]}}}}}'
        with self.assertRaises(pb_utils.TritonModelException):
            pb_utils.load_model(model_name=self.model_name, config=wrong_config)

        for version in ["2", "3"]:
            self.assertTrue(
                pb_utils.is_model_ready(
                    model_name=self.model_name, model_version=version
                )
            )

        config = (
            '{"backend":"onnxruntime", "version_policy":{"specific":{"versions":[2]}}}'
        )
        pb_utils.load_model(self.model_name, config=config)

        self.assertTrue(pb_utils.is_model_ready(self.model_name, "2"))
        self.assertFalse(pb_utils.is_model_ready(self.model_name, "3"))

    def test_load_with_file_override(self):
        self.assertFalse(pb_utils.is_model_ready(self.model_name))
        pb_utils.load_model(self.model_name)
        self.assertTrue(pb_utils.is_model_ready(self.model_name))

        override_name = "override_model"
        config = '{"backend":"onnxruntime"}'
        with open("models/onnx_int32_int32_int32/3/model.onnx", "rb") as file:
            data = file.read()
        files = {"file:1/model.onnx": data}

        with self.assertRaises(pb_utils.TritonModelException):
            pb_utils.load_model(self.model_name, "", files)

        pb_utils.load_model(model_name=override_name, config=config, files=files)

        self.assertFalse(pb_utils.is_model_ready(self.model_name, "1"))
        self.assertTrue(pb_utils.is_model_ready(self.model_name, "3"))

        self.assertTrue(pb_utils.is_model_ready(override_name, "1"))
        self.assertFalse(pb_utils.is_model_ready(override_name, "3"))

        pb_utils.load_model(self.model_name, config, files)

        self.assertTrue(pb_utils.is_model_ready(self.model_name, "1"))
        self.assertFalse(pb_utils.is_model_ready(self.model_name, "3"))

        self.assertTrue(pb_utils.is_model_ready(override_name, "1"))
        self.assertFalse(pb_utils.is_model_ready(override_name, "3"))


class TritonPythonModel:
    def initialize(self, args):

        test = unittest.main("model", exit=False)
        self.result = test.result.wasSuccessful()

    def execute(self, requests):
        responses = []
        for _ in requests:
            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor(
                            "OUTPUT0", np.array([self.result], dtype=np.float16)
                        )
                    ]
                )
            )
        return responses
