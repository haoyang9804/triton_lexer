import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):

        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", in_0.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses
