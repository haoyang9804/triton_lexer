import argparse
import sys
from builtins import range

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
    elements = 10

    client = client_util.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose)

    input_data = []
    input_data.append(np.arange(start=1, stop=1 + elements, dtype=np.float32))
    input_data.append(np.array([2] * elements, dtype=np.float32))

    inputs = []
    for i in range(len(input_data)):
        inputs.append(
            client_util.InferInput(
                "INPUT__{}".format(i),
                input_data[0].shape,
                np_to_triton_dtype(input_data[0].dtype),
            )
        )
        inputs[i].set_data_from_numpy(input_data[i])

    results = client.infer(model_name, inputs)

    output_data = results.as_numpy("OUTPUT__0")
    if output_data is None:
        print("error: expected 'OUTPUT__0'")
        sys.exit(1)

    for i in range(elements):
        print(
            str(i)
            + ": "
            + str(input_data[0][i])
            + " % "
            + str(input_data[1][i])
            + " = "
            + str(output_data[i])
        )
        if (input_data[0][i] % input_data[1][i]) != output_data[i]:
            print("error: incorrect value")
            sys.exit(1)
