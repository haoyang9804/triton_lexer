import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self._index = 0
        self._dtypes = [np.bytes_, np.object_]

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor_0 = pb_utils.Tensor(
                "OUTPUT0", in_0.as_numpy().astype(self._dtypes[self._index])
            )
            self._index += 1
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses
