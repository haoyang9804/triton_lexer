

























import json
import threading
import time

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

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        

        for request in requests:
            thread = threading.Thread(
                target=self.response_thread,
                args=(
                    request.get_response_sender(),
                    pb_utils.get_input_tensor_by_name(request, "IN").as_numpy(),
                ),
            )
            thread.daemon = True
            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1
            thread.start()

        return None

    def response_thread(self, response_sender, in_value):
        infer_request = pb_utils.InferenceRequest(
            model_name="square_int32",
            requested_output_names=["OUT"],
            inputs=[pb_utils.Tensor("IN", in_value)],
        )
        infer_responses = infer_request.exec(decoupled=True)

        response_count = 0
        for infer_response in infer_responses:
            if len(infer_response.output_tensors()) > 0:
                output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUT")
                if infer_response.has_error():
                    response = pb_utils.InferenceResponse(
                        error=infer_response.error().message()
                    )
                    response_sender.send(
                        response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                elif np.any(in_value != output0.as_numpy()):
                    error_message = (
                        "BLS Request input and BLS response output do not match."
                        f" {in_value} != {output0.as_numpy()}"
                    )
                    response = pb_utils.InferenceResponse(error=error_message)
                    response_sender.send(
                        response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    output_tensors = [pb_utils.Tensor("OUT", output0.as_numpy())]
                    response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                    response_sender.send(response)

            response_count += 1

        if in_value != response_count - 1:
            error_message = "Expected {} responses, got {}".format(
                in_value, len(infer_responses) - 1
            )
            response = pb_utils.InferenceResponse(error=error_message)
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
        else:
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        inflight_threads = True
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
            if inflight_threads:
                time.sleep(0.1)
