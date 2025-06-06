import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def execute(self, requests):
        request = requests[0]

        input = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()
        max_response_count = pb_utils.get_input_tensor_by_name(
            request, "MAX_RESPONSE_COUNT"
        ).as_numpy()[0]
        delay = pb_utils.get_input_tensor_by_name(request, "DELAY").as_numpy()[0]
        ignore_cancel = pb_utils.get_input_tensor_by_name(
            request, "IGNORE_CANCEL"
        ).as_numpy()[0]
        response_sender = request.get_response_sender()

        sent = 0
        while True:
            if not ignore_cancel and request.is_cancelled():
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        message="request has been cancelled",
                        code=pb_utils.TritonError.CANCELLED,
                    )
                )
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                break

            output = pb_utils.Tensor("OUTPUT", np.array([input[0]], np.int32))
            response = pb_utils.InferenceResponse(output_tensors=[output])

            if sent + 1 == max_response_count:
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                break
            else:
                response_sender.send(response)
                sent += 1
                time.sleep(delay / 1000)

        return None
