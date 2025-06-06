import json

import torch
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):

        self.model_config = json.loads(args["model_config"])

        self.input0_config = pb_utils.get_input_config_by_name(
            self.model_config, "INPUT0"
        )
        self.output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT0"
        )

    def validate_bf16_tensor(self, tensor, tensor_config):

        dtype = tensor_config["data_type"]
        if dtype != "TYPE_BF16":
            raise Exception(f"Expected a BF16 tensor, but got {dtype} instead.")

        try:
            _ = tensor.as_numpy()
        except pb_utils.TritonModelException as e:
            expected_error = "tensor dtype is bf16 and cannot be converted to numpy"
            assert expected_error in str(e).lower()
        else:
            raise Exception("Expected BF16 conversion to numpy to fail")

    def execute(self, requests):

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")

            bf16_dlpack = input_tensor.to_dlpack()

            torch_tensor = torch.utils.dlpack.from_dlpack(bf16_dlpack)

            output_tensor = pb_utils.Tensor.from_dlpack(
                "OUTPUT0", torch.utils.dlpack.to_dlpack(torch_tensor)
            )
            responses.append(pb_utils.InferenceResponse([output_tensor]))

            self.validate_bf16_tensor(input_tensor, self.input0_config)
            self.validate_bf16_tensor(output_tensor, self.output0_config)

        return responses
