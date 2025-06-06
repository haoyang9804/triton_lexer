import json
import os

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        input0 = {"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [4]}
        input1 = {"name": "INPUT1", "data_type": "TYPE_FP32", "dims": [4]}
        output0 = {"name": "OUTPUT0", "data_type": "TYPE_FP32", "dims": [4]}
        output1 = {"name": "OUTPUT1", "data_type": "TYPE_FP32", "dims": [4]}

        auto_complete_model_config.set_max_batch_size(0)
        auto_complete_model_config.add_input(input0)
        auto_complete_model_config.add_input(input1)
        auto_complete_model_config.add_output(output0)
        auto_complete_model_config.add_output(output1)

        return auto_complete_model_config

    def initialize(self, args):

        model_path = os.path.join(args["model_repository"], args["model_version"])
        self.indicator_file = os.path.join(model_path, "indicator")
        if not os.path.exists(self.indicator_file):
            with open(self.indicator_file, "x") as f:
                pass
            raise Exception("failing first load attempt")

        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

    def finalize(self):

        os.remove(self.indicator_file)

    def execute(self, requests):

        pass
