import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []

        for request in requests:
            res_params_tensor = pb_utils.get_input_tensor_by_name(
                request, "RESPONSE_PARAMETERS"
            ).as_numpy()
            res_params_str = str(res_params_tensor[0][0], encoding="utf-8")
            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array([[res_params_str]], dtype=np.object_)
            )
            try:
                res_params = json.loads(res_params_str)

                if isinstance(res_params, dict):
                    res_params_new = {}
                    for key, value in res_params.items():
                        if isinstance(key, str) and key.isdigit():
                            key = int(key)
                        res_params_new[key] = value
                    res_params = res_params_new

                response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor], parameters=res_params
                )

                res_params_set = {}
                if response.parameters() != "":
                    res_params_set = json.loads(response.parameters())
                if res_params_set != res_params:
                    raise Exception("Response parameters set differ from provided")
            except Exception as e:
                error = pb_utils.TritonError(
                    message=str(e), code=pb_utils.TritonError.INVALID_ARG
                )
                response = pb_utils.InferenceResponse(error=error)

            responses.append(response)

        return responses
