import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []

        for request in requests:
            input0_np = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            input1_np = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()

            output0_np = input0_np + input1_np

            output_tensors = [
                pb_utils.Tensor("OUTPUT0", output0_np.astype(np.int32)),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors))

        return responses
