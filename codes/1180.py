import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def execute(self, requests):
        responses = []

        for request in requests:
            passed = True

            res_params_tensor = pb_utils.get_input_tensor_by_name(
                request, "RESPONSE_PARAMETERS"
            ).as_numpy()
            res_params_str = str(res_params_tensor[0][0], encoding="utf-8")
            res_params = json.loads(res_params_str)
            bls_input_tensor = pb_utils.Tensor("RESPONSE_PARAMETERS", res_params_tensor)
            bls_req = pb_utils.InferenceRequest(
                model_name="response_parameters",
                inputs=[bls_input_tensor],
                requested_output_names=["OUTPUT"],
            )
            bls_res = bls_req.exec()
            bls_res_params_str = bls_res.parameters()
            bls_res_params = (
                json.loads(bls_res_params_str) if bls_res_params_str != "" else {}
            )
            passed = passed and bls_res_params == res_params

            res_params_decoupled_tensor = pb_utils.get_input_tensor_by_name(
                request, "RESPONSE_PARAMETERS_DECOUPLED"
            ).as_numpy()
            res_params_decoupled_str = str(
                res_params_decoupled_tensor[0][0], encoding="utf-8"
            )
            res_params_decoupled = json.loads(res_params_decoupled_str)
            bls_decoupled_input_tensor = pb_utils.Tensor(
                "RESPONSE_PARAMETERS", res_params_decoupled_tensor
            )
            bls_decoupled_req = pb_utils.InferenceRequest(
                model_name="response_parameters_decoupled",
                inputs=[bls_decoupled_input_tensor],
                requested_output_names=["OUTPUT"],
            )
            bls_decoupled_res = bls_decoupled_req.exec(decoupled=True)
            for bls_decoupled_r in bls_decoupled_res:
                if len(bls_decoupled_r.output_tensors()) == 0:
                    break
                bls_decoupled_r_params_str = bls_decoupled_r.parameters()
                bls_decoupled_r_params = (
                    json.loads(bls_decoupled_r_params_str)
                    if bls_decoupled_r_params_str != ""
                    else {}
                )
                passed = passed and bls_decoupled_r_params in res_params_decoupled
                res_params_decoupled.remove(bls_decoupled_r_params)
            passed = passed and len(res_params_decoupled) == 0

            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array([[str(passed)]], dtype=np.object_)
            )
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses
