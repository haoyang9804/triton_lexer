import argparse
import sys
from builtins import range

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.utils as utils
import tritonclient.utils.shared_memory as shm

FLAGS = None


def infer_and_validata(use_shared_memory, orig_input0_data, orig_input1_data):
    if use_shared_memory:
        input0_data = orig_input0_data
        input1_data = orig_input1_data
        byte_size = input0_data.size * input0_data.itemsize
        inputs[0].set_shared_memory("input0_data", byte_size)
        inputs[1].set_shared_memory("input1_data", byte_size)
        outputs[0].set_shared_memory("output0_data", byte_size)
        outputs[1].set_shared_memory("output1_data", byte_size)
    else:
        input0_data = orig_input0_data
        input1_data = orig_input1_data * 2
        inputs[0].set_data_from_numpy(np.expand_dims(input0_data, axis=0))
        inputs[1].set_data_from_numpy(np.expand_dims(input1_data, axis=0))
        outputs[0].unset_shared_memory()
        outputs[1].unset_shared_memory()

    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    output0 = results.get_output("OUTPUT0")
    if output0 is not None:
        if use_shared_memory:
            if protocol == "grpc":
                output0_data = shm.get_contents_as_numpy(
                    shm_op0_handle,
                    utils.triton_to_np_dtype(output0.datatype),
                    output0.shape,
                )
            else:
                output0_data = shm.get_contents_as_numpy(
                    shm_op0_handle,
                    utils.triton_to_np_dtype(output0["datatype"]),
                    output0["shape"],
                )
        else:
            output0_data = results.as_numpy("OUTPUT0")
    else:
        print("OUTPUT0 is missing in the response.")
        sys.exit(1)

    output1 = results.get_output("OUTPUT1")
    if output1 is not None:
        if use_shared_memory:
            if protocol == "grpc":
                output1_data = shm.get_contents_as_numpy(
                    shm_op1_handle,
                    utils.triton_to_np_dtype(output1.datatype),
                    output1.shape,
                )
            else:
                output1_data = shm.get_contents_as_numpy(
                    shm_op1_handle,
                    utils.triton_to_np_dtype(output1["datatype"]),
                    output1["shape"],
                )
        else:
            output1_data = results.as_numpy("OUTPUT1")
    else:
        print("OUTPUT1 is missing in the response.")
        sys.exit(1)

    if use_shared_memory:
        print("\n\n======== SHARED_MEMORY ========\n")
    else:
        print("\n\n======== NO_SHARED_MEMORY ========\n")
    for i in range(16):
        print(
            str(input0_data[i])
            + " + "
            + str(input1_data[i])
            + " = "
            + str(output0_data[0][i])
        )
        print(
            str(input0_data[i])
            + " - "
            + str(input1_data[i])
            + " = "
            + str(output1_data[0][i])
        )
        if (input0_data[i] + input1_data[i]) != output0_data[0][i]:
            print("shm infer error: incorrect sum")
            sys.exit(1)
        if (input0_data[i] - input1_data[i]) != output1_data[0][i]:
            print("shm infer error: incorrect difference")
            sys.exit(1)
    print("\n======== END ========\n\n")


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
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="HTTP",
        help="Protocol (HTTP/gRPC) used to communicate with "
        + "the inference service. Default is HTTP.",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )

    FLAGS = parser.parse_args()

    protocol = FLAGS.protocol.lower()

    try:
        if protocol == "grpc":

            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose
            )
        else:

            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose
            )
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    triton_client.unregister_system_shared_memory()
    triton_client.unregister_cuda_shared_memory()

    model_name = "simple"
    model_version = ""

    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input1_data = np.ones(shape=16, dtype=np.int32)

    input_byte_size = input0_data.size * input0_data.itemsize
    output_byte_size = input_byte_size

    shm_op0_handle = shm.create_shared_memory_region(
        "output0_data", "/output0_simple", output_byte_size
    )
    shm_op1_handle = shm.create_shared_memory_region(
        "output1_data", "/output1_simple", output_byte_size
    )

    triton_client.register_system_shared_memory(
        "output0_data", "/output0_simple", output_byte_size
    )
    triton_client.register_system_shared_memory(
        "output1_data", "/output1_simple", output_byte_size
    )

    shm_ip0_handle = shm.create_shared_memory_region(
        "input0_data", "/input0_simple", input_byte_size
    )
    shm_ip1_handle = shm.create_shared_memory_region(
        "input1_data", "/input1_simple", input_byte_size
    )

    shm.set_shared_memory_region(shm_ip0_handle, [input0_data])
    shm.set_shared_memory_region(shm_ip1_handle, [input1_data])

    triton_client.register_system_shared_memory(
        "input0_data", "/input0_simple", input_byte_size
    )
    triton_client.register_system_shared_memory(
        "input1_data", "/input1_simple", input_byte_size
    )

    inputs = []
    if protocol == "grpc":
        inputs.append(grpcclient.InferInput("INPUT0", [1, 16], "INT32"))

        inputs.append(grpcclient.InferInput("INPUT1", [1, 16], "INT32"))
    else:
        inputs.append(httpclient.InferInput("INPUT0", [1, 16], "INT32"))

        inputs.append(httpclient.InferInput("INPUT1", [1, 16], "INT32"))

    outputs = []
    if protocol == "grpc":
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT1"))
    else:
        outputs.append(httpclient.InferRequestedOutput("OUTPUT0"))
        outputs.append(httpclient.InferRequestedOutput("OUTPUT1"))

    infer_and_validata(True, input0_data, input1_data)

    infer_and_validata(False, input0_data, input1_data)

    infer_and_validata(True, input0_data, input1_data)

    triton_client.unregister_system_shared_memory()
    shm.destroy_shared_memory_region(shm_ip0_handle)
    shm.destroy_shared_memory_region(shm_ip1_handle)
    shm.destroy_shared_memory_region(shm_op0_handle)
    shm.destroy_shared_memory_region(shm_op1_handle)
