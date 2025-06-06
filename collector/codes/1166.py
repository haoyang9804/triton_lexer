import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):

        logger = pb_utils.Logger
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            logger.log_info(f"Request timeout: {request.timeout()}")
            time.sleep(5)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
