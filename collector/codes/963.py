import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack


class TritonPythonModel:
    def initialize(self, args):

        self.device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        self.model = (
            torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
            .to(self.device)
            .eval()
        )

    def execute(self, requests):

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            result = self.model(
                torch.as_tensor(input_tensor.as_numpy(), device=self.device)
            )
            out_tensor = pb_utils.Tensor.from_dlpack("OUTPUT0", to_dlpack(result))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
