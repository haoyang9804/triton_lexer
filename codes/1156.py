import json
import threading
import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self._logger = pb_utils.Logger
        self._model_config = json.loads(args["model_config"])
        self._using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self._model_config
        )

    def execute(self, requests):
        processed_requests = []
        for request in requests:
            delay_tensor = pb_utils.get_input_tensor_by_name(
                request, "EXECUTE_DELAY"
            ).as_numpy()
            delay = delay_tensor[0][0]
            if self._using_decoupled:
                processed_requests.append(
                    {"response_sender": request.get_response_sender(), "delay": delay}
                )
            else:
                processed_requests.append({"request": request, "delay": delay})
        if self._using_decoupled:
            return self._execute_decoupled(processed_requests)
        return self._execute_processed_requests(processed_requests)

    def _execute_processed_requests(self, processed_requests):
        responses = []
        for processed_request in processed_requests:
            error = pb_utils.TritonError(message="not cancelled")
            object_to_check_cancelled = None
            if "response_sender" in processed_request:
                object_to_check_cancelled = processed_request["response_sender"]
            elif "request" in processed_request:
                object_to_check_cancelled = processed_request["request"]
            delay = processed_request["delay"]
            time_elapsed = 0.0
            while time_elapsed < delay:
                time.sleep(1)
                time_elapsed += 1.0
                if object_to_check_cancelled.is_cancelled():
                    self._logger.log_info(
                        "[execute_cancel] Request cancelled at "
                        + str(time_elapsed)
                        + " s"
                    )
                    error = pb_utils.TritonError(
                        message="cancelled", code=pb_utils.TritonError.CANCELLED
                    )
                    break
                self._logger.log_info(
                    "[execute_cancel] Request not cancelled at "
                    + str(time_elapsed)
                    + " s"
                )
            responses.append(pb_utils.InferenceResponse(error=error))
        return responses

    def _execute_decoupled(self, processed_requests):
        def response_thread(execute_processed_requests, processed_requests):
            time.sleep(2)
            responses = execute_processed_requests(processed_requests)
            for i in range(len(responses)):
                response_sender = processed_requests[i]["response_sender"]
                response_sender.send(responses[i])
                response_sender.send(
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

        thread = threading.Thread(
            target=response_thread,
            args=(self._execute_processed_requests, processed_requests),
        )
        thread.daemon = True
        thread.start()
        return None
