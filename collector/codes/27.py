from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np


model_name = "model_pipeline"

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.random.rand(3, 640, 640).astype(np.float32)
    inputs = [
        httpclient.InferInput(
            "IMAGE", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        )
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("CLASSIFICATION"),
        httpclient.InferRequestedOutput("BBOXES"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    result = response.get_response()
    print("OUTPUT0 ({})".format(response.as_numpy("CLASSIFICATION")))
    print("OUTPUT0 ({})".format(response.as_numpy("BBOXES")))
