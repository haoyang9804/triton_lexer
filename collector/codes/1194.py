import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack


class TritonPythonModel:
    def execute(self, requests):

        assert len(requests) == 5
        responses = []
        for i, request in enumerate(requests):

            output = torch.ones(i + 1, dtype=torch.float32, device="cuda")
            output = output * (i + 1)
            output_pb_tensor = pb_utils.Tensor.from_dlpack("OUTPUT", to_dlpack(output))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_pb_tensor]
            )
            responses.append(inference_response)
        return responses
