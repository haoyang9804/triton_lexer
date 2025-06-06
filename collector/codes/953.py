import argparse
import sys

import numpy as np
import tritonclient.grpc as grpcclient

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

    FLAGS = parser.parse_args()
    try:

        channel_args = [
            ("grpc.max_send_message_length", 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024),
            ("grpc.keepalive_time_ms", 2**31 - 1),
            ("grpc.keepalive_timeout_ms", 20000),
            ("grpc.keepalive_permit_without_calls", False),
            ("grpc.http2.max_pings_without_data", 2),
            ("grpc.dns_enable_srv_queries", 1),
        ]

        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose, channel_args=channel_args
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
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

    results = triton_client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs, headers={"test": "1"}
    )

    output0_data = results.as_numpy("OUTPUT0")
    output1_data = results.as_numpy("OUTPUT1")

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

    print("PASS: CustomArgs")
