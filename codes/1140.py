import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []

        for request in requests:
            num_params = int(
                pb_utils.get_input_tensor_by_name(
                    request, "NUMBER_PARAMETERS"
                ).as_numpy()[0]
            )
            params = json.loads(request.parameters())

            if num_params == 0:

                response = json.dumps(params)
                response_tensors = [
                    pb_utils.Tensor(
                        "PARAMETERS_AGGREGATED", np.array([response], dtype=np.object_)
                    )
                ]
            else:

                params["bool_" + str(num_params)] = bool(num_params)
                params["int_" + str(num_params)] = num_params
                params["str_" + str(num_params)] = str(num_params)

                bls_request_tensor = pb_utils.Tensor(
                    "NUMBER_PARAMETERS", np.array([num_params - 1], dtype=np.ubyte)
                )
                bls_request = pb_utils.InferenceRequest(
                    model_name="bls_parameters",
                    inputs=[bls_request_tensor],
                    requested_output_names=["PARAMETERS_AGGREGATED"],
                    parameters=params,
                )
                bls_response = bls_request.exec()
                response_tensors = bls_response.output_tensors()

            inference_response = pb_utils.InferenceResponse(
                output_tensors=response_tensors
            )
            responses.append(inference_response)

        return responses
