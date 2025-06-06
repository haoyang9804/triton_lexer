import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        output0_dtype = self.output0_dtype

        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")

            out_0 = in_0.as_numpy()

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )

            request.set_release_flags(pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE)

            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
