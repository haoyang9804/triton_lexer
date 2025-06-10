import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        error_code_map = {
            "UNKNOWN": pb_utils.TritonError.UNKNOWN,
            "INTERNAL": pb_utils.TritonError.INTERNAL,
            "NOT_FOUND": pb_utils.TritonError.NOT_FOUND,
            "INVALID_ARG": pb_utils.TritonError.INVALID_ARG,
            "UNAVAILABLE": pb_utils.TritonError.UNAVAILABLE,
            "UNSUPPORTED": pb_utils.TritonError.UNSUPPORTED,
            "ALREADY_EXISTS": pb_utils.TritonError.ALREADY_EXISTS,
            "CANCELLED": pb_utils.TritonError.CANCELLED,
        }

        responses = []

        for request in requests:
            err_code_tensor = pb_utils.get_input_tensor_by_name(
                request, "ERROR_CODE"
            ).as_numpy()
            err_code_str = str(err_code_tensor[0][0], encoding="utf-8")
            if err_code_str in error_code_map:
                error = pb_utils.TritonError(
                    message=("error code: " + err_code_str),
                    code=error_code_map[err_code_str],
                )
            else:
                error = pb_utils.TritonError("unrecognized error code: " + err_code_str)
            responses.append(pb_utils.InferenceResponse(error=error))

        return responses
