import os
from sys import platform
from typing import Tuple

import numpy as np
import torch
import tritonclient.http as httpclient
from tritonclient import utils
from tritonclient.utils import InferenceServerException

if "linux" in platform:

    import tritonclient.utils.shared_memory as shm

MODEL_PATH = os.path.join("model_repository", "mnist_cnn", "1", "model.pt")
TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "localhost:9000")


class BasePredictor:

    def predict(self, _: torch.FloatTensor) -> int:

        raise NotImplementedError


class Predictor(BasePredictor):

    def __init__(self, model_path: str = MODEL_PATH) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model_path = model_path

    def predict(self, image: torch.FloatTensor) -> int:

        logits = self.model(image.to(self.device))
        prediction = torch.argmax(logits).detach().cpu().item()
        return int(prediction)


class PredictorTriton(BasePredictor):

    def __init__(
        self,
        model_name: str = "mnist_cnn",
        input_info: Tuple[str, Tuple[int, ...]] = ("input__0", (1, 1, 28, 28)),
        output_name: str = "output__0",
        type_name: str = "FP32",
    ) -> None:

        self.url = TRITON_SERVER_URL
        self.model_name = model_name
        input_name, input_shape = input_info
        self.output_name = output_name

        self.client = httpclient.InferenceServerClient(url=self.url, verbose=False)
        self.inputs = [httpclient.InferInput(input_name, input_shape, type_name)]
        self.outputs = [httpclient.InferRequestedOutput(output_name, binary_data=False)]

    def predict(self, image: torch.FloatTensor) -> int:

        try:
            self.inputs[0].set_data_from_numpy(image.numpy(), binary_data=False)
            results = self.client.infer(
                self.model_name, self.inputs, outputs=self.outputs
            )
            _ = results.get_response()
            output = results.as_numpy(self.output_name)
            prediction = output.argmax()

        except InferenceServerException:
            return -1

        return int(prediction)


class PredictorTritonShm(BasePredictor):

    def __init__(
        self,
        model_name: str = "mnist_cnn",
        input_info: Tuple[str, Tuple[int, ...]] = ("input__0", (1, 1, 28, 28)),
        output_info: Tuple[str, Tuple[int, ...]] = ("output__0", (1, 10)),
        type_name: str = "FP32",
    ) -> None:

        url = TRITON_SERVER_URL
        self.model_name = model_name
        self.output_name, output_shape = output_info
        self.input_name, input_shape = input_info

        self.client = httpclient.InferenceServerClient(url=url, verbose=False)
        self.__init_shared_memory(input_shape, output_shape, type_name)

    def __init_shared_memory(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        type_name: str,
    ) -> None:

        self.client.unregister_system_shared_memory()
        self.client.unregister_cuda_shared_memory()

        input_data = np.zeros(input_shape, dtype=np.float32)
        output_data = np.zeros(output_shape, dtype=np.float32)
        intput_byte_size = input_data.size * input_data.itemsize
        output_byte_size = output_data.size * output_data.itemsize

        self.shm_op_handle = shm.create_shared_memory_region(
            "output_data", f"/output_{self.model_name}", output_byte_size
        )
        self.client.register_system_shared_memory(
            "output_data", f"/output_{self.model_name}", output_byte_size
        )
        shm_ip_handle = shm.create_shared_memory_region(
            "input_data", f"/input_{self.model_name}", intput_byte_size
        )
        shm.set_shared_memory_region(shm_ip_handle, [input_data])
        self.client.register_system_shared_memory(
            "input_data", f"/input_{self.model_name}", intput_byte_size
        )

        self.inputs = [httpclient.InferInput(self.input_name, input_shape, type_name)]
        self.inputs[-1].set_shared_memory("input_data", intput_byte_size)
        self.outputs = [
            httpclient.InferRequestedOutput(self.output_name, binary_data=True)
        ]
        self.outputs[-1].set_shared_memory("output_data", output_byte_size)

    def predict(self, image: torch.FloatTensor) -> int:

        self.inputs[-1].set_data_from_numpy(image.numpy().astype(np.float32))
        results = self.client.infer(self.model_name, self.inputs, outputs=self.outputs)
        output = results.get_output(self.output_name)
        if output is None:
            return -1

        output = shm.get_contents_as_numpy(
            self.shm_op_handle,
            utils.triton_to_np_dtype(output["datatype"]),
            output["shape"],
        )
        prediction = output.argmax()
        return int(prediction)
