import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [{"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [1]}]
        outputs = [
            {"name": "key", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "value", "data_type": "TYPE_STRING", "dims": [-1]},
        ]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        for input in inputs:
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)
        return auto_complete_model_config

    def execute(self, requests):

        responses = []
        for request in requests:
            parameters = json.loads(request.parameters())
            keys = []
            values = []
            for key, value in parameters.items():
                keys.append(key)
                values.append(value)
            key_output = pb_utils.Tensor("key", np.asarray(keys, dtype=object))
            value_output = pb_utils.Tensor("value", np.asarray(values, dtype=object))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[key_output, value_output]
            )
            responses.append(inference_response)

        return responses
