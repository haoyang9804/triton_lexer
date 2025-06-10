import sys
import triton_python_backend_utils as pb_utils

sys.path.append("../../")


class TritonPythonModel:

    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "text_input")
            out_tensor_0 = pb_utils.Tensor("text_output", in_0.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses
