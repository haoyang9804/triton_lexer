import unittest

import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack


class PBBLSONNXWarmupTest(unittest.TestCase):
    def test_onnx_output_mem_type(self):
        input0_np = np.random.randn(*[16])
        input0_np = input0_np.astype(np.float32)
        input1_np = np.random.randn(*[16])
        input1_np = input1_np.astype(np.float32)
        input0 = pb_utils.Tensor("INPUT0", input0_np)
        input1 = pb_utils.Tensor("INPUT1", input1_np)
        infer_request = pb_utils.InferenceRequest(
            model_name="onnx_nobatch_float32_float32_float32",
            inputs=[input0, input1],
            requested_output_names=["OUTPUT0", "OUTPUT1"],
        )

        infer_response = infer_request.exec()

        self.assertFalse(infer_response.has_error())

        output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
        output1 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT1")

        self.assertIsNotNone(output0)
        self.assertIsNotNone(output1)

        self.assertFalse(output0.is_cpu())
        self.assertFalse(output1.is_cpu())

        expected_output_0 = input0.as_numpy() - input1.as_numpy()
        expected_output_1 = input0.as_numpy() + input1.as_numpy()

        output0 = from_dlpack(output0.to_dlpack()).to("cpu").cpu().detach().numpy()
        output1 = from_dlpack(output1.to_dlpack()).to("cpu").cpu().detach().numpy()

        self.assertTrue(np.all(output0 == expected_output_0))
        self.assertTrue(np.all(output1 == expected_output_1))


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
