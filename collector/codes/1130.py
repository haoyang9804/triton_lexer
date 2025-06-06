import asyncio

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    async def _execute_a_request(self, request):
        input_tensor = pb_utils.get_input_tensor_by_name(
            request, "WAIT_SECONDS"
        ).as_numpy()
        bls_input_tensor = pb_utils.Tensor("WAIT_SECONDS", input_tensor)
        bls_request = pb_utils.InferenceRequest(
            model_name="async_execute_decouple",
            inputs=[bls_input_tensor],
            requested_output_names=["DUMMY_OUT"],
        )
        bls_responses = await bls_request.async_exec(decoupled=True)
        response_sender = request.get_response_sender()
        for bls_response in bls_responses:
            bls_output_tensor = pb_utils.get_output_tensor_by_name(
                bls_response, "DUMMY_OUT"
            ).as_numpy()
            output_tensor = pb_utils.Tensor("DUMMY_OUT", bls_output_tensor)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            response_sender.send(response)
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    async def execute(self, requests):
        async_futures = []
        for request in requests:
            async_future = self._execute_a_request(request)
            async_futures.append(async_future)
        await asyncio.gather(*async_futures)
        return None
