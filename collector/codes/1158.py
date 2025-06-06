import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):

        responses = []

        i = 0
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "IN")
            out_tensor = pb_utils.Tensor("OUT", input_tensor.as_numpy())
            if i == 0:
                error = pb_utils.TritonError("An error occurred during execution")
                responses.append(pb_utils.InferenceResponse([out_tensor], error))
            elif i == 1:
                responses.append(pb_utils.InferenceResponse([out_tensor]))
            elif i == 2:
                error = pb_utils.TritonError("An error occurred during execution")
                responses.append(pb_utils.InferenceResponse(error=error))
            i += 1

        return responses
