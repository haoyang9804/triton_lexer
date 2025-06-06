import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.decoupled = self.model_config.get("model_transaction_policy", {}).get(
            "decoupled"
        )

    def execute(self, requests):
        if self.decoupled:
            return self.exec_decoupled(requests)
        else:
            return self.exec(requests)

    def exec(self, requests):
        responses = []
        for request in requests:
            params = json.loads(request.parameters())
            rep_count = params["REPETITION"] if "REPETITION" in params else 1

            input_np = pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy()
            stream_np = pb_utils.get_input_tensor_by_name(request, "STREAM").as_numpy()
            stream = stream_np.flatten()[0]
            if stream:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "STREAM only supported in decoupled mode"
                        )
                    )
                )
            else:
                out_tensor = pb_utils.Tensor(
                    "TEXT", np.repeat(input_np, rep_count, axis=1)
                )
                responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def exec_decoupled(self, requests):
        for request in requests:
            params = json.loads(request.parameters())
            rep_count = params["REPETITION"] if "REPETITION" in params else 1
            fail_last = params["FAIL_LAST"] if "FAIL_LAST" in params else False
            delay = params["DELAY"] if "DELAY" in params else None
            output_0_dim = params["OUTPUT_0_DIM"] if "OUTPUT_0_DIM" in params else False

            sender = request.get_response_sender()
            input_np = pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy()
            stream_np = pb_utils.get_input_tensor_by_name(request, "STREAM").as_numpy()
            out_value = np.array([]) if output_0_dim else input_np
            out_tensor = pb_utils.Tensor("TEXT", out_value)
            response = pb_utils.InferenceResponse([out_tensor])

            stream = stream_np.flatten()[0]
            if stream:
                for _ in range(rep_count):
                    if delay is not None:
                        time.sleep(delay)
                    if not sender.is_cancelled():
                        sender.send(response)
                    else:
                        break
                sender.send(
                    (
                        None
                        if not fail_last
                        else pb_utils.InferenceResponse(
                            error=pb_utils.TritonError("An Error Occurred")
                        )
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )

            else:
                sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
        return None
