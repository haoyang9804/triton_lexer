import os
import sys
import time

import triton_python_backend_utils as pb_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from util import get_delay, inc_count


class TritonPythonModel:
    def initialize(self, args):
        inc_count("initialize")
        self._sleep("initialize")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        self._sleep("infer")
        return responses

    def finalize(self):
        inc_count("finalize")

    def _sleep(self, kind):
        delay = get_delay(kind)
        if delay > 0:
            time.sleep(delay)
