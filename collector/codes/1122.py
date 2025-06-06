import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def execute(self, requests):
        for request in requests:
            request.get_response_sender().send(
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )

        return None
