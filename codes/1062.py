import argparse
from builtins import range

import numpy as np
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
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of model.")
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        required=True,
        help="Number of asynchronous requests to launch.",
    )

    FLAGS = parser.parse_args()

    model_name = FLAGS.model

    client = httpclient.InferenceServerClient(
        FLAGS.url, verbose=FLAGS.verbose, concurrency=FLAGS.num_requests
    )

    requests = []

    tensor_size = [1, 5 * 1024 * 1024]
    input_data = np.random.randn(*tensor_size).astype(np.float32)

    inputs = [
        httpclient.InferInput(
            "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    ]
    inputs[0].set_data_from_numpy(input_data)

    for i in range(FLAGS.num_requests):
        requests.append(client.async_infer(model_name, inputs))
        print("Sent request %d" % i, flush=True)

    for i in range(len(requests)):
        requests[i].get_result()
        print("Received result %d" % i, flush=True)
