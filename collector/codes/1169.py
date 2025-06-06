

























import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                .format(
                    args["model_name"]
                )
            )

        
        in_config = pb_utils.get_input_config_by_name(model_config, "IN")

        
        in_shape = in_config["dims"]
        if (len(in_shape) != 1) or (in_shape[0] != 1):
            raise pb_utils.TritonModelException(
                .format(
                    args["model_name"], in_shape
                )
            )
        if in_config["data_type"] != "TYPE_INT32":
            raise pb_utils.TritonModelException(
                .format(
                    args["model_name"], in_config["data_type"]
                )
            )

        
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        
        out_shape = out_config["dims"]
        if (len(out_shape) != 1) or (out_shape[0] != 1):
            raise pb_utils.TritonModelException(
                .format(
                    args["model_name"], out_shape
                )
            )
        if out_config["data_type"] != "TYPE_INT32":
            raise pb_utils.TritonModelException(
                .format(
                    args["model_name"], out_config["data_type"]
                )
            )

        self.remaining_response = 0
        self.reset_flag = True

    def execute(self, requests):
        for request in requests:
            in_input = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()

            if self.reset_flag:
                self.remaining_response = in_input[0]
                self.reset_flag = False

            response_sender = request.get_response_sender()

            self.remaining_response -= 1

            out_output = pb_utils.Tensor(
                "OUT", np.array([self.remaining_response], np.int32)
            )
            response = pb_utils.InferenceResponse(output_tensors=[out_output])

            if self.remaining_response <= 0:
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
            else:
                request.set_release_flags(
                    pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
                )
                response_sender.send(response)

        return None
