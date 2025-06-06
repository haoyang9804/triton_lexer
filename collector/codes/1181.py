import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        for request in requests:
            res_params_tensor = pb_utils.get_input_tensor_by_name(
                request, "RESPONSE_PARAMETERS"
            ).as_numpy()
            res_params_str = str(res_params_tensor[0][0], encoding="utf-8")
            response_sender = request.get_response_sender()
            try:
                res_params = json.loads(res_params_str)
                for r_params in res_params:
                    output_tensor = pb_utils.Tensor(
                        "OUTPUT", np.array([[json.dumps(r_params)]], dtype=np.object_)
                    )
                    response = pb_utils.InferenceResponse(
                        output_tensors=[output_tensor], parameters=r_params
                    )

                    r_params_set = {}
                    if response.parameters() != "":
                        r_params_set = json.loads(response.parameters())
                    if r_params_set != r_params:
                        raise Exception("Response parameters set differ from provided")

                    response_sender.send(response)
            except Exception as e:
                error = pb_utils.TritonError(
                    message=str(e), code=pb_utils.TritonError.INVALID_ARG
                )
                response = pb_utils.InferenceResponse(error=error)
                response_sender.send(response)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        return None
