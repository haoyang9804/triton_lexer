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
        "-u", "--url", type=str, required=False, help="Inference server URL."
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

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print(
            'unexpected protocol "{}", expects "http" or "grpc"'.format(FLAGS.protocol)
        )
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    with client_util.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose) as client:
        for model_name, np_dtype, shape in (
            ("identity_int32", np.int32, [0]),
            ("identity_int32", np.int32, [7]),
        ):

            if np_dtype != object:
                input_data = (16384 * np.random.randn(*shape)).astype(np_dtype)
            else:
                in0 = 16384 * np.ones(shape, dtype="int")
                in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                input_data = in0n.reshape(in0.shape)
            inputs = [
                client_util.InferInput(
                    "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data)

            results = client.infer(model_name, inputs)
            print(results)

            output_data = results.as_numpy("OUTPUT0")
            if output_data is None:
                print("error: expected 'OUTPUT0'")
                sys.exit(1)

            if np_dtype == object:
                output_data = np.char.decode(output_data)

            if not np.array_equal(output_data, input_data):
                print(
                    "error: expected output {} to match input {}".format(
                        output_data, input_data
                    )
                )
                sys.exit(1)
