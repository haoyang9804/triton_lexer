import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self.sleep = True

    def execute(self, requests):
        if self.sleep:
            time.sleep(50)
            self.sleep = False
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
