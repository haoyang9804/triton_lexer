import argparse
import sys

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

FLAGS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="http",
        help='Protocol ("http"/"grpc") used to '
        + 'communicate with inference service. Default is "http".',
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of model.")

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print(
            'unexpected protocol "{}", expects "http" or "grpc"'.format(FLAGS.protocol)
        )
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    model_name = FLAGS.model

    client = client_util.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose)

    input_data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    box_data = np.array([[1, 1, 2, 3, 4]]).astype(np.float32)

    inputs = []
    inputs.append(
        client_util.InferInput(
            "INPUT__0", input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    )
    inputs[0].set_data_from_numpy(input_data)
    inputs.append(
        client_util.InferInput(
            "INPUT__1", box_data.shape, np_to_triton_dtype(box_data.dtype)
        )
    )
    inputs[1].set_data_from_numpy(box_data)

    results = client.infer(model_name, inputs)

    output_data = results.as_numpy("OUTPUT__0")
    if output_data is None:
        print("error: expected 'OUTPUT__0'")
        sys.exit(1)

    if output_data.shape != (1, 3, 5, 5):
        print("error: incorrect shape " + str(output_data.shape) + "for 'OUTPUT__0'")
        sys.exit(1)
