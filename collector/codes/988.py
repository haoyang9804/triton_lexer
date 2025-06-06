import unittest

import numpy as np
import triton_python_backend_utils as pb_utils


class ArgumentValidationTest(unittest.TestCase):
    def test_infer_request_args(self):

        inputs = [pb_utils.Tensor("INPUT0", np.asarray([1, 2], dtype=np.int32))]
        model_name = "my_model"
        requested_output_names = ["my_output"]

        with self.assertRaises(pb_utils.TritonModelException) as e:
            pb_utils.InferenceRequest(
                inputs=[None],
                model_name=model_name,
                requested_output_names=requested_output_names,
            )

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceRequest(
                inputs=None,
                model_name=model_name,
                requested_output_names=requested_output_names,
            )

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceRequest(
                model_name=None,
                inputs=inputs,
                requested_output_names=requested_output_names,
            )

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceRequest(
                requested_output_names=[None], inputs=inputs, model_name=model_name
            )

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceRequest(
                requested_output_names=None, inputs=inputs, model_name=model_name
            )

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceRequest(
                requested_output_names=requested_output_names,
                inputs=inputs,
                model_name=model_name,
                correlation_id=None,
            )

        infer_request_test = pb_utils.InferenceRequest(
            requested_output_names=requested_output_names,
            inputs=inputs,
            model_name=model_name,
            correlation_id=5,
        )
        self.assertIsInstance(infer_request_test.correlation_id(), int)
        self.assertEqual(infer_request_test.correlation_id(), 5)

        infer_request_test = pb_utils.InferenceRequest(
            requested_output_names=requested_output_names,
            inputs=inputs,
            model_name=model_name,
            correlation_id="test_str_id-5",
        )
        self.assertIsInstance(infer_request_test.correlation_id(), str)
        self.assertEqual(infer_request_test.correlation_id(), "test_str_id-5")

        infer_request_test = pb_utils.InferenceRequest(
            requested_output_names=requested_output_names,
            inputs=inputs,
            model_name=model_name,
        )
        self.assertIsInstance(infer_request_test.correlation_id(), int)
        self.assertEqual(infer_request_test.correlation_id(), 0)

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceRequest(
                requested_output_names=requested_output_names,
                inputs=inputs,
                model_name=model_name,
                request_id=None,
            )

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceRequest(
                requested_output_names=requested_output_names,
                inputs=inputs,
                model_name=model_name,
                model_version=None,
            )

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceRequest(
                requested_output_names=requested_output_names,
                inputs=inputs,
                model_name=model_name,
                flags=None,
            )

        pb_utils.InferenceRequest(
            requested_output_names=[], inputs=[], model_name=model_name
        )

    def test_infer_response_args(self):
        outputs = [pb_utils.Tensor("OUTPUT0", np.asarray([1, 2], dtype=np.int32))]

        with self.assertRaises(pb_utils.TritonModelException) as e:
            pb_utils.InferenceResponse(output_tensors=[None])

        with self.assertRaises(TypeError) as e:
            pb_utils.InferenceResponse(output_tensors=None)

        pb_utils.InferenceResponse(output_tensors=[])
        pb_utils.InferenceResponse(outputs)

    def test_tensor_args(self):
        np_array = np.asarray([1, 2], dtype=np.int32)

        with self.assertRaises(TypeError) as e:
            pb_utils.Tensor(None, np_array)

        with self.assertRaises(TypeError) as e:
            pb_utils.Tensor("OUTPUT0", None)

        with self.assertRaises(pb_utils.TritonModelException) as e:
            pb_utils.Tensor.from_dlpack("OUTPUT0", None)

        with self.assertRaises(pb_utils.TritonModelException) as e:
            pb_utils.Tensor.from_dlpack("", None)

        with self.assertRaises(TypeError) as e:
            pb_utils.Tensor("", None)

    def test_log_args(self):
        logger = pb_utils.Logger

        with self.assertRaises(TypeError) as e:
            logger.log("Invalid Level", None)

        with self.assertRaises(TypeError) as e:
            logger.log("Invalid Level", 1)

        with self.assertRaises(TypeError) as e:
            logger.log_info(None)

        with self.assertRaises(TypeError) as e:
            logger.log_warn(None)

        with self.assertRaises(TypeError) as e:
            logger.log_error(None)

        with self.assertRaises(TypeError) as e:
            logger.log_verbose(None)

        logger.log("Level unspecified")


class TritonPythonModel:

    def execute(self, requests):
        responses = []
        for _ in requests:

            test = unittest.main("model", exit=False)
            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor(
                            "OUTPUT0",
                            np.array([test.result.wasSuccessful()], dtype=np.float16),
                        )
                    ]
                )
            )
        return responses
