import os
import unittest

import numpy as np
import triton_python_backend_utils as pb_utils


class PBBLSMemoryTest(unittest.TestCase):
    def setUp(self):
        self._is_decoupled = True if os.environ["BLS_KIND"] == "decoupled" else False

    def _send_identity_tensor(self, size, is_decoupled):
        tensor_size = [1, size]
        input0_np = np.random.randn(*tensor_size)
        input0 = pb_utils.Tensor("INPUT0", input0_np.astype(np.float32))
        infer_request = pb_utils.InferenceRequest(
            model_name="identity_fp32",
            inputs=[input0],
            requested_output_names=["OUTPUT0"],
        )

        if is_decoupled:
            infer_responses = infer_request.exec(decoupled=True)
            infer_response = next(infer_responses)
            with self.assertRaises(StopIteration):
                next(infer_responses)
        else:
            infer_response = infer_request.exec()

        return input0_np, infer_response

    def test_bls_out_of_memory(self):
        tensor_size = 256 * 1024 * 1024
        input0_np, infer_response = self._send_identity_tensor(
            tensor_size, self._is_decoupled
        )
        out_of_memory_message = "Failed to increase the shared memory pool size for key"

        if infer_response.has_error():
            self.assertIn(out_of_memory_message, infer_response.error().message())
        else:
            self.assertFalse(infer_response.has_error())
            output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
            self.assertIsNotNone(output0)
            self.assertTrue(np.allclose(output0.as_numpy(), input0_np))

        tensor_size = 50 * 1024 * 1024
        for _ in range(4):
            input0_np, infer_response = self._send_identity_tensor(
                tensor_size, self._is_decoupled
            )
            if infer_response.has_error():
                self.assertIn(out_of_memory_message, infer_response.error().message())
            else:
                self.assertFalse(infer_response.has_error())
                output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
                self.assertIsNotNone(output0)
                self.assertTrue(np.allclose(output0.as_numpy(), input0_np))


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
