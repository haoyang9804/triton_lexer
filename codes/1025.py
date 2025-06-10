import argparse
import sys

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument("-i", "--protocol", type=str, required=True)
    FLAGS = parser.parse_args()

    if FLAGS.protocol == "grpc":
        client_type = grpcclient
    else:
        client_type = httpclient

    try:
        triton_client = client_type.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "simple"

    inputs = []
    outputs = []
    inputs.append(client_type.InferInput("INPUT0", [1, 16], "INT32"))
    inputs.append(client_type.InferInput("INPUT1", [1, 16], "INT32"))

    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.ones(shape=(1, 16), dtype=np.int32)

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs.append(client_type.InferRequestedOutput("OUTPUT0"))
    outputs.append(client_type.InferRequestedOutput("OUTPUT1"))

    triton_client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs, request_id="1"
    )
