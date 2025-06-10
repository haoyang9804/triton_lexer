import json
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils


def get_valid_param_value(param, default_value=""):
    value = param.get("string_value", "")
    return default_value if value.startswith("${") or value == "" else value


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.output_config = pb_utils.get_output_config_by_name(
            model_config, "text_output"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            self.output_config["data_type"]
        )
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(model_config)
        self.logger = pb_utils.Logger

    def create_triton_tensors(self, index):
        x = "bla" + str(index)
        output = [x.encode("utf8")]
        np_output = np.array(output).astype(self.output_dtype)
        seq_idx = np.array([[0]]).astype(np.int32)

        t1 = pb_utils.Tensor("text_output", np_output)
        t2 = pb_utils.Tensor("sequence_index", seq_idx)
        tensors = [t1, t2]
        return tensors

    def create_triton_response(self, index):
        tensors = self.create_triton_tensors(index)
        return pb_utils.InferenceResponse(output_tensors=tensors)

    def execute(self, requests):
        responses = []
        for request in requests:
            if self.decoupled:
                response_sender = request.get_response_sender()
            try:
                for index in range(0, 1):
                    triton_response = self.create_triton_response(index)
                    if self.decoupled:
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

                if self.decoupled:
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )

            except Exception:
                self.logger.log_error(traceback.format_exc())
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(traceback.format_exc()),
                )

                if self.decoupled:
                    response_sender.send(error_response)
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    responses.append(error_response)

        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses

    def finalize(self):
        self.logger.log_info("Cleaning up...")
