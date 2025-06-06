import json


import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):

        self.model_config = model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):

        output_dtype = self.output_dtype

        responses = []

        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")

            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            out_0 = in_0.as_numpy() + in_1.as_numpy()

            out_tensor_0 = pb_utils.Tensor("OUTPUT", out_0.astype(output_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")
