import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):

        responses = []
        for request in requests:
            input0_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input1_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            if input0_tensor is not None:
                input0_numpy = input0_tensor.as_numpy()
            else:
                input0_numpy = np.array([5], dtype=np.int32)

            if input1_tensor is not None:
                input1_numpy = input1_tensor.as_numpy()
            else:
                input1_numpy = np.array([5], dtype=np.int32)

            output0_tensor = pb_utils.Tensor("OUTPUT0", input0_numpy + input1_numpy)
            output1_tensor = pb_utils.Tensor("OUTPUT1", input0_numpy - input1_numpy)
            responses.append(
                pb_utils.InferenceResponse([output0_tensor, output1_tensor])
            )

        return responses
