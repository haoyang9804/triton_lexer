import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for _ in requests:
            SHAPE = (0,)

            pytorch_tensor = torch.ones(SHAPE, dtype=torch.float32)

            device = torch.device("cuda:0")
            pytorch_tensor = pytorch_tensor.to(device)

            dlpack_tensor = to_dlpack(pytorch_tensor)
            pb_tensor = pb_utils.Tensor.from_dlpack("OUTPUT", dlpack_tensor)

            inference_response = pb_utils.InferenceResponse(output_tensors=[pb_tensor])
            responses.append(inference_response)

        return responses
