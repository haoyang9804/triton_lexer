import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.sequences = {}
        self.decoupled = self.model_config.get("model_transaction_policy", {}).get(
            "decoupled"
        )

    def get_next_sequence_output_tensor(self, request):
        sid = request.correlation_id()
        flags = request.flags()
        if flags == pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START:
            if sid in self.sequences:
                raise pb_utils.TritonModelException(
                    "Can't start a new sequence with existing ID"
                )
            self.sequences[sid] = [1]
        else:
            if sid not in self.sequences:
                raise pb_utils.TritonModelException(
                    "Need START flag for a sequence ID that doesn't already exist."
                )

            last = self.sequences[sid][-1]
            self.sequences[sid].append(last + 1)

        output = self.sequences[sid][-1]
        output = np.array([output])
        out_tensor = pb_utils.Tensor("OUTPUT0", output.astype(np.int32))
        return out_tensor

    def execute(self, requests):
        if self.decoupled:
            return self.execute_decoupled(requests)
        else:
            return self.execute_non_decoupled(requests)

    def execute_non_decoupled(self, requests):
        responses = []
        for request in requests:
            output_tensor = self.get_next_sequence_output_tensor(request)
            response = pb_utils.InferenceResponse([output_tensor])
            responses.append(response)
        return responses

    def execute_decoupled(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            output_tensor = self.get_next_sequence_output_tensor(request)

            for _ in range(3):
                response = pb_utils.InferenceResponse([output_tensor])
                sender.send(response)

            sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        return None

    def finalize(self):
        print(f"Cleaning up. Final sequences stored: {self.sequences}")
