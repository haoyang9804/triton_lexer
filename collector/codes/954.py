import argparse
import queue
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

FLAGS = None


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


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
        help="Inference server URL and it gRPC port. Default is localhost:8001.",
    )

    FLAGS = parser.parse_args()

    model_name = "repeat_int32"
    model_version = ""
    repeat_count = 10
    data_offset = 100
    delay_time = 1000
    wait_time = 1000

    input_data = np.arange(
        start=data_offset, stop=data_offset + repeat_count, dtype=np.int32
    )
    delay_data = (np.ones([repeat_count], dtype=np.uint32)) * delay_time
    wait_data = np.array([wait_time], dtype=np.uint32)

    inputs = []
    inputs.append(grpcclient.InferInput("IN", [repeat_count], "INT32"))
    inputs[-1].set_data_from_numpy(input_data)
    inputs.append(grpcclient.InferInput("DELAY", [repeat_count], "UINT32"))
    inputs[-1].set_data_from_numpy(delay_data)
    inputs.append(grpcclient.InferInput("WAIT", [1], "UINT32"))
    inputs[-1].set_data_from_numpy(wait_data)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("OUT"))

    result_list = []

    user_data = UserData()

    with grpcclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose
    ) as triton_client:
        try:

            triton_client.start_stream(callback=partial(callback, user_data))

            triton_client.async_stream_infer(
                model_name=model_name, inputs=inputs, outputs=outputs
            )
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

        recv_count = 0
        while recv_count < repeat_count:
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                print(data_item)
                sys.exit(1)
            else:
                result_list.append(data_item.as_numpy("OUT"))
            recv_count = recv_count + 1

    expected_data = data_offset
    for i in range(len(result_list)):
        if len(result_list[i]) != 1:
            print(
                "unexpected number of elements in the output, expected 1, got {}".format(
                    len(result_list[i])
                )
            )
            sys.exit(1)
        print("{} : {}".format(result_list[i][0], expected_data))
        if result_list[i][0] != expected_data:
            print("mismatch in the results")
            sys.exit(1)
        expected_data += 1
    print("PASS: Decoupled API")
