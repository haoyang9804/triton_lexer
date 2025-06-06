import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    async def execute(self, requests):
        request = requests[0]
        wait_secs = pb_utils.get_input_tensor_by_name(
            request, "WAIT_SECONDS"
        ).as_numpy()[0]
        response_num = pb_utils.get_input_tensor_by_name(
            request, "RESPONSE_NUM"
        ).as_numpy()[0]
        output_tensors = [
            pb_utils.Tensor("WAIT_SECONDS", np.array([wait_secs], np.float32)),
            pb_utils.Tensor("RESPONSE_NUM", np.array([1], np.uint8)),
        ]

        time.sleep(wait_secs.item())
        response_sender = request.get_response_sender()
        for i in range(response_num):
            response = pb_utils.InferenceResponse(output_tensors)
            if i != response_num - 1:
                response_sender.send(response)
            else:
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

        return None
