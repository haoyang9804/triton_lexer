import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def execute(self, requests):
        responses = []

        for _ in requests:
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError("An Error Occurred")
                )
            )

        return responses
