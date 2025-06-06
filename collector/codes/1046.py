import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):

        responses = []
        for _ in requests:

            out_0 = np.array([1], dtype=np.float32)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0)
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))

        return responses
