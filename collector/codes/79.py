import os
import tritonclient.grpc.aio as grpcclient
import numpy as np

from .utils import decode_img, normalize_img


class ImageClassifier:

    def __init__(self) -> None:

        url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
        self.triton_client = grpcclient.InferenceServerClient(url=url)
        self.outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]

    async def predict(
        self,
        img: str,
        height: int,
        width: int,
        model_name: str = "mnist_cnn",
        mean: float = 0.1307,
        std: float = 0.3081,
    ) -> int:

        img = decode_img(img)
        img = normalize_img(img, mean, std)
        img = img.reshape(1, 1, height, width)

        inputs = [grpcclient.InferInput("INPUT__0", img.shape, "FP32")]
        inputs[0].set_data_from_numpy(img.astype(np.float32))
        results = await self.triton_client.infer(
            model_name=model_name, inputs=inputs, outputs=self.outputs
        )

        output = results.as_numpy("OUTPUT__0")
        top_1 = output.argmax()
        return top_1
