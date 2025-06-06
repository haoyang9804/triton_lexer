import argparse

import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *


def main(model_name):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    url = "http://images.cocodataset.org/val2017/000000161642.jpg"
    image = np.asarray(Image.open(requests.get(url, stream=True).raw)).astype(
        np.float32
    )
    image = np.expand_dims(image, axis=0)

    input_tensors = [httpclient.InferInput("image", image.shape, datatype="FP32")]
    input_tensors[0].set_data_from_numpy(image)

    outputs = [httpclient.InferRequestedOutput("last_hidden_state")]

    query_response = client.infer(
        model_name=model_name, inputs=input_tensors, outputs=outputs
    )

    last_hidden_state = query_response.as_numpy("last_hidden_state")
    print(last_hidden_state.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="Select between enemble_model and python_vit"
    )
    args = parser.parse_args()
    main(args.model_name)
