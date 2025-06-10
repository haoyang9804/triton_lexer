import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        new_shape = [10, 2, 6, 5, 11]
        shape_reorder = [1, 0, 4, 2, 3]
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_numpy = input_tensor.as_numpy()
            output0 = pb_utils.Tensor("OUTPUT0", input_numpy.reshape(new_shape))

            output1 = pb_utils.Tensor("OUTPUT1", input_numpy.T)
            output2 = pb_utils.Tensor(
                "OUTPUT2", np.transpose(input_numpy, shape_reorder)
            )
            responses.append(pb_utils.InferenceResponse([output0, output1, output2]))
        return responses
