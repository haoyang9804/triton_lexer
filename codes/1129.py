import asyncio

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    async def execute(self, requests):
        processed_requests = []
        async_tasks = []
        for request in requests:
            wait_secs_tensors = pb_utils.get_input_tensor_by_name(
                request, "WAIT_SECONDS"
            ).as_numpy()
            for wait_secs_tensor in wait_secs_tensors:
                wait_secs = wait_secs_tensor[0]
                if wait_secs < 0:
                    self.raise_value_error(requests)
                async_tasks.append(asyncio.create_task(asyncio.sleep(wait_secs)))
            processed_requests.append(
                {
                    "response_sender": request.get_response_sender(),
                    "batch_size": wait_secs_tensors.shape[0],
                }
            )

        await asyncio.gather(*async_tasks)

        for p_req in processed_requests:
            response_sender = p_req["response_sender"]
            batch_size = p_req["batch_size"]

            output_tensors = pb_utils.Tensor(
                "DUMMY_OUT", np.array([0 for i in range(batch_size)], np.float32)
            )
            response = pb_utils.InferenceResponse(output_tensors=[output_tensors])
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )

        return None

    def raise_value_error(self, requests):

        for request in requests:
            response_sender = request.get_response_sender()
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        raise ValueError("wait_secs cannot be negative")
