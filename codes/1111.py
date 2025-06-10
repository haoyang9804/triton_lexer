import random
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient

if __name__ == "__main__":

    client_timeout = 1
    url = "localhost:8001"

    try:
        triton_client = grpcclient.InferenceServerClient(url=url)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = "identity_fp32"

    inputs = []

    input_data = np.array(
        [random.random() for i in range(50)], dtype=np.float32
    ).reshape(1, -1)
    model_input = grpcclient.InferInput(
        name="INPUT0", datatype="FP32", shape=input_data.shape
    )
    model_input.set_data_from_numpy(input_data)
    inputs.append(model_input)

    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)

    user_data = []

    for _ in range(1000):
        triton_client.async_infer(
            model_name=model_name,
            inputs=inputs,
            callback=partial(callback, user_data),
            client_timeout=client_timeout,
        )

    time_out = 20
    while (len(user_data) == 0) and time_out > 0:
        time_out = time_out - 1
        time.sleep(1)

    print("results: ", len(user_data))
