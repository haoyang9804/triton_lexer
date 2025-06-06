import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
