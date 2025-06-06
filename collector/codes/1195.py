import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "IN")
            out_tensor = pb_utils.Tensor("OUT", input_tensor.as_numpy())
            lorem_ipsum
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
