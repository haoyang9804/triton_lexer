import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self._index = 0
        self._dtypes = [np.bytes_, np.object_]

    def execute(self, requests):

        responses = []
        for _ in requests:
            if self._index == 0:
                out_tensor_0 = pb_utils.Tensor(
                    "OUTPUT0", np.array(["123456"], dtype=self._dtypes[0])
                )
            elif self._index == 1:
                out_tensor_0 = pb_utils.Tensor(
                    "OUTPUT0", np.array([], dtype=self._dtypes[1])
                )
            elif self._index == 2:
                out_tensor_0 = pb_utils.Tensor(
                    "OUTPUT0", np.array(["123456"], dtype=self._dtypes[0])
                )
            elif self._index == 3:
                out_tensor_0 = pb_utils.Tensor(
                    "OUTPUT0", np.array([], dtype=self._dtypes[1])
                )
            self._index += 1
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses
