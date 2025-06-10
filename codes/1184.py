import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):

        if len(requests) != 1:
            pb_utils.Logger.log_error(f"Unexpected request length: {len(requests)}")
            raise Exception("Test FAILED")

        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", in_tensor.as_numpy())
            response = pb_utils.InferenceResponse([out_tensor])
            response_sender = request.get_response_sender()
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
            test_passed = False
            try:
                response_sender.send(response)
            except Exception as e:
                pb_utils.Logger.log_info(f"Raised exception: {e}")
                if (
                    str(e)
                    == "Unable to send response. Response sender has been closed."
                ):
                    test_passed = True
            finally:
                if not test_passed:
                    pb_utils.Logger.log_error("Expected exception not raised")
                    raise Exception("Test FAILED")
            pb_utils.Logger.log_info("Test Passed")
        return None
