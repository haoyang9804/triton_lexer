import asyncio

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    async def execute(self, requests):
        max_sum = (
            pb_utils.get_input_tensor_by_name(requests[0], "MAX_SUM").as_numpy().flat[0]
        )
        input = pb_utils.get_input_tensor_by_name(requests[0], "INPUT")
        ignore_cancel = pb_utils.get_input_tensor_by_name(requests[0], "IGNORE_CANCEL")
        delay = pb_utils.Tensor("DELAY", np.array([50], dtype=np.int32))
        max_response_count = pb_utils.Tensor(
            "MAX_RESPONSE_COUNT", np.array([20], dtype=np.int32)
        )

        infer_request = pb_utils.InferenceRequest(
            model_name="response_sender_until_cancelled",
            inputs=[input, max_response_count, delay, ignore_cancel],
            requested_output_names=["OUTPUT"],
        )

        response_stream = await infer_request.async_exec(decoupled=True)

        is_cancelled = False
        error = None
        response_sum = 0
        for infer_response in response_stream:
            if infer_response.has_error():
                if infer_response.error().code() == pb_utils.TritonError.CANCELLED:
                    is_cancelled = True
                else:
                    error = infer_response.error()
                break

            out = pb_utils.get_output_tensor_by_name(
                infer_response, "OUTPUT"
            ).as_numpy()[0]

            response_sum += out
            if response_sum >= max_sum:
                response_stream.cancel()

        responses = [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("SUM", np.array([response_sum], dtype=np.int32)),
                    pb_utils.Tensor(
                        "IS_CANCELLED", np.array([is_cancelled], dtype=np.bool_)
                    ),
                ],
                error=error,
            )
        ]

        return responses
