import time

import triton_python_backend_utils as pb_utils


time.sleep(5)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "IN")
            out_tensor = pb_utils.Tensor("OUT", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def finalize(self):
        pass
