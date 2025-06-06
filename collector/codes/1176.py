import locale
import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        input = {"name": "INPUT", "data_type": "TYPE_FP32", "dims": [1]}
        output = {"name": "OUTPUT", "data_type": "TYPE_FP32", "dims": [1]}

        auto_complete_model_config.set_max_batch_size(0)
        auto_complete_model_config.add_input(input)
        auto_complete_model_config.add_output(output)

        return auto_complete_model_config

    def initialize(self, args):
        import torch

        self.model_config = args["model_config"]

        os.system("/bin/bash --help")
        print(
            f"Python version is {sys.version_info.major}.{sys.version_info.minor}, NumPy version is {np.version.version}, and PyTorch version is {torch.__version__}",
            flush=True,
        )
        print(f"Locale is {locale.getlocale()}", flush=True)

    def execute(self, requests):

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
