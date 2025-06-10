import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        self.accumulator = np.zeros(1)
        self.max_batch_size = model_config["max_batch_size"]

    def execute(self, requests):

        output_dtype = self.output_dtype

        responses = []
        for request in requests:
            input_tensor = (
                pb_utils.get_input_tensor_by_name(request, "INPUT")
                .as_numpy()
                .astype(np.int32)
            )
            start_tensor = (
                pb_utils.get_input_tensor_by_name(request, "START")
                .as_numpy()
                .astype(np.int32)
            )
            ready_tensor = (
                pb_utils.get_input_tensor_by_name(request, "READY")
                .as_numpy()
                .astype(np.int32)
            )

            if self.max_batch_size == 0:
                tmp = np.where(
                    np.equal(start_tensor, 1),
                    input_tensor,
                    np.add(self.accumulator, input_tensor),
                )
                newacc = np.where(np.equal(ready_tensor, 1), tmp, self.accumulator)
                self.accumulator = newacc
                out_tensor = pb_utils.Tensor(
                    "OUTPUT", self.accumulator.astype(output_dtype)
                )
            else:
                tmp = np.where(
                    np.equal(ready_tensor, 1),
                    np.add(start_tensor, input_tensor),
                    np.zeros(np.shape(input_tensor), dtype=output_dtype),
                )
                out_tensor = pb_utils.Tensor("OUTPUT", tmp.astype(output_dtype))

            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
