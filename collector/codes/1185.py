import json

import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

    def execute(self, requests):

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        for request in requests:
            response_sender = request.get_response_sender()
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            out_0, out_1 = (
                in_0.as_numpy() + in_1.as_numpy(),
                in_0.as_numpy() - in_1.as_numpy(),
            )

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_1.astype(output1_dtype))
            response_sender.send(
                pb_utils.InferenceResponse([out_tensor_0, out_tensor_1])
            )
            response_sender.close()

        return None
