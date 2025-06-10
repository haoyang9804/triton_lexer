import argparse
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

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
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-t",
        "--client-timeout",
        type=float,
        required=False,
        default=None,
        help="Client timeout in seconds. Default is None.",
    )

    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose
        )
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = "simple"

    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput("INPUT0", [1, 16], "INT32"))
    inputs.append(grpcclient.InferInput("INPUT1", [1, 16], "INT32"))

    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.ones(shape=(1, 16), dtype=np.int32)

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))
    outputs.append(grpcclient.InferRequestedOutput("OUTPUT1"))

    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)

    user_data = []

    triton_client.async_infer(
        model_name=model_name,
        inputs=inputs,
        callback=partial(callback, user_data),
        outputs=outputs,
        client_timeout=FLAGS.client_timeout,
    )

    time_out = 10
    while (len(user_data) == 0) and time_out > 0:
        time_out = time_out - 1
        time.sleep(1)

    if len(user_data) == 1:

        if type(user_data[0]) == InferenceServerException:
            print(user_data[0])
            sys.exit(1)

        output0_data = user_data[0].as_numpy("OUTPUT0")
        output1_data = user_data[0].as_numpy("OUTPUT1")
        for i in range(16):
            print(
                str(input0_data[0][i])
                + " + "
                + str(input1_data[0][i])
                + " = "
                + str(output0_data[0][i])
            )
            print(
                str(input0_data[0][i])
                + " - "
                + str(input1_data[0][i])
                + " = "
                + str(output1_data[0][i])
            )
            if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
                print("sync infer error: incorrect sum")
                sys.exit(1)
            if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
                print("sync infer error: incorrect difference")
                sys.exit(1)
        print("PASS: Async infer")
