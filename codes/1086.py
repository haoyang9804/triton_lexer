import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):

        assert len(requests) == 1
        delay = 0
        request = requests[0]
        responses = []

        delay_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
        delay_as_numpy = delay_tensor.as_numpy()
        delay = float(delay_as_numpy[0][0])

        out_tensor = pb_utils.Tensor("OUTPUT0", delay_as_numpy)
        responses.append(pb_utils.InferenceResponse([out_tensor]))

        time.sleep(delay)
        return responses
