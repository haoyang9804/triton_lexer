import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:

            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            infer_request = pb_utils.InferenceRequest(
                model_name="identity",
                requested_output_names=["OUTPUT0"],
                inputs=[input0],
            )
            infer_response = infer_request.exec()

            if infer_response.has_error():
                raise pb_utils.TritonModelException(infer_response.error().message())

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
                ]
            )
            responses.append(inference_response)

        return responses
