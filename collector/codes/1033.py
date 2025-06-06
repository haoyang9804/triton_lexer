import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []

        for request in requests:
            json_string = (
                pb_utils.get_input_tensor_by_name(request, "EXPECTED_HEADERS")
                .as_numpy()[0]
                .decode("utf-8")
            )
            expected_headers = json.loads(json_string)

            success = True
            if request.parameters() != "":
                parameters = json.loads(request.parameters())
                for key, value in expected_headers.items():
                    if key in parameters:
                        if parameters[key] != value:
                            success = False
                    else:
                        success = False

            test_success = pb_utils.Tensor(
                "TEST_SUCCESS", np.array([success], dtype=bool)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[test_success]
            )
            responses.append(inference_response)

        return responses
