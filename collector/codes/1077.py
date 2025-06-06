import argparse

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
        "-m", "--model-name", type=str, required=True, help="Name of model"
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

    FLAGS = parser.parse_args()

    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print(
            'unexpected protocol "{}", expects "http" or "grpc"'.format(FLAGS.protocol)
        )
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    client = client_util.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose)

    batch_size = 1

    tmp_str = "abc\0def"
    input0_data = np.array([tmp_str], dtype=object)

    input_name = "INPUT0"
    output_name = "OUTPUT0"

    if "libtorch" in FLAGS.model_name:
        input_name = "INPUT__0"
        output_name = "OUTPUT__0"

    inputs = [
        client_util.InferInput(
            input_name, input0_data.shape, np_to_triton_dtype(np.object_)
        )
    ]
    inputs[0].set_data_from_numpy(input0_data)

    results = client.infer(FLAGS.model_name, inputs)

    output0_data = results.as_numpy(output_name)

    print(input0_data, "?=?", output0_data)
    assert np.equal(input0_data.astype(np.bytes_), output0_data).all()
