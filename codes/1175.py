import json
import os

import triton_python_backend_utils as pb_utils

_ADD_SUB_ARGS_FILENAME = "model.json"


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):

        inputs = [
            {"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [4]},
            {"name": "INPUT1", "data_type": "TYPE_FP32", "dims": [4]},
        ]
        outputs = [{"name": "OUTPUT", "data_type": "TYPE_FP32", "dims": [4]}]

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

        return auto_complete_model_config

    def initialize(self, args):

        self.model_config = model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        engine_args_filepath = os.path.join(
            pb_utils.get_model_dir(), _ADD_SUB_ARGS_FILENAME
        )
        assert os.path.isfile(
            engine_args_filepath
        ), f"'{_ADD_SUB_ARGS_FILENAME}' containing add sub model args must be provided in '{pb_utils.get_model_dir()}'"

        with open(engine_args_filepath) as file:
            self.add_sub_config = json.load(file)

        assert (
            "operation" in self.add_sub_config
        ), f"Missing required key 'operation' in {_ADD_SUB_ARGS_FILENAME}"

        extra_keys = set(self.add_sub_config.keys()) - {"operation"}
        assert (
            not extra_keys
        ), f"Unsupported keys are provided in {_ADD_SUB_ARGS_FILENAME}: {', '.join(extra_keys)}"

        assert self.add_sub_config["operation"] in [
            "add",
            "sub",
        ], f"'operation' value must be 'add' or 'sub' in {_ADD_SUB_ARGS_FILENAME}"

        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):

        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            if self.add_sub_config["operation"] == "add":
                out = in_0.as_numpy() + in_1.as_numpy()
            else:
                out = in_0.as_numpy() - in_1.as_numpy()

            out_tensor = pb_utils.Tensor("OUTPUT", out.astype(self.output_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")
