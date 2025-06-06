import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils


def check_init_args(args):
    expected_args = {
        "model_name": "init_args",
        "model_instance_name": "init_args_0_0",
        "model_instance_kind": "CPU",
        "model_instance_device_id": "0",
        "model_version": "1",
    }
    is_win = sys.platform == "win32"
    triton_dir = os.getenv(
        "TRITON_DIR", "c:\\tritonserver" if is_win else "/opt/tritonserver"
    )
    repo_path = triton_dir + "/qa/L0_backend_python/models/init_args"
    expected_args["model_repository"] = (
        repo_path.replace("/", "\\") if is_win else repo_path
    )

    for arg in expected_args:
        if args[arg] != expected_args[arg]:
            raise pb_utils.TritonModelException(
                arg
                + ' does not contain correct value. Expected "'
                + expected_args[arg]
                + ", got "
                + args[arg]
            )


class TritonPythonModel:
    def initialize(self, args):
        self.args = args
        check_init_args(self.args)

    def execute(self, requests):

        keys = [
            "model_config",
            "model_instance_kind",
            "model_instance_name",
            "model_instance_device_id",
            "model_repository",
            "model_version",
            "model_name",
        ]

        correct_keys = 0
        for key in keys:
            if key in list(self.args):
                correct_keys += 1

        responses = []
        for _ in requests:
            out_args = pb_utils.Tensor(
                "OUT", np.array([correct_keys], dtype=np.float32)
            )
            responses.append(pb_utils.InferenceResponse([out_args]))
        return responses
