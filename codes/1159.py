import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):

        self.inf_count = 1

    def execute(self, requests):

        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "IN")
            out_tensor = pb_utils.Tensor("OUT", input_tensor.as_numpy())
            if self.inf_count % 2:

                responses.append(pb_utils.InferenceResponse([out_tensor]))
            else:

                error = pb_utils.TritonError("An error occurred during execution")
                responses.append(pb_utils.InferenceResponse([out_tensor], error))
            self.inf_count += 1

        return responses
