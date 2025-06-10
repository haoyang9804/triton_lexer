import sys

sys.path.append("../common")

import argparse
import time
from multiprocessing import Process, shared_memory

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


def crashing_client(
    model_name, dtype, tensor_shape, shm_name, triton_client, input_name="INPUT0"
):
    in0 = np.random.random(tensor_shape).astype(dtype)
    if "libtorch" in model_name:
        input_name = "INPUT__0"
    inputs = [
        grpcclient.InferInput(input_name, tensor_shape, np_to_triton_dtype(dtype)),
    ]
    inputs[0].set_data_from_numpy(in0)

    while True:
        existing_shm = shared_memory.SharedMemory(shm_name)
        count = np.ndarray((1,), dtype=np.int32, buffer=existing_shm.buf)
        count[0] += 1
        existing_shm.close()
        results = triton_client.infer(model_name, inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--trial",
        type=str,
        required=True,
        help="Set trial for the crashing client",
    )
    FLAGS = parser.parse_args()
    trial = FLAGS.trial

    dtype = np.float32
    model_name = tu.get_zero_model_name(trial, 1, dtype)
    tensor_shape = (1,) if "nobatch" in trial else (1, 1)

    triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=True)

    shm = shared_memory.SharedMemory(create=True, size=8)
    count = np.ndarray((1,), dtype=np.int32, buffer=shm.buf)
    count[0] = 0

    p = Process(
        target=crashing_client,
        name="crashing_client",
        args=(
            model_name,
            dtype,
            tensor_shape,
            shm.name,
            triton_client,
        ),
    )

    p.start()

    time.sleep(3)
    p.terminate()

    p.join()

    print("request_count:", count[0])

    shm.close()
    shm.unlink()

    if not triton_client.is_server_live():
        sys.exit(1)

    sys.exit(0)
