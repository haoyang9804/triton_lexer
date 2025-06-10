import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        pass

    def finalize(self):
        print("Cleaning up...")
        input0_np = np.random.randint(3, size=1, dtype=np.int32)
        input0 = pb_utils.Tensor("IN", input0_np)
        infer_request = pb_utils.InferenceRequest(
            model_name="square_int32", inputs=[input0], requested_output_names=["OUT"]
        )
        infer_responses = infer_request.exec(decoupled=True)
