import argparse
import threading
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from print_utils import Display


def client1_callback(display, event, result, error):
    if error:
        raise error

    display.update_top()
    if result.get_response().parameters.get("triton_final_response").bool_param:
        event.set()


def client2_callback(display, event, result, error):
    if error:
        raise error

    display.update_bottom()
    if result.get_response().parameters.get("triton_final_response").bool_param:
        event.set()


def run_inferences(url, model_name, display, max_tokens):

    client1 = grpcclient.InferenceServerClient(url)
    client2 = grpcclient.InferenceServerClient(url)

    inputs0 = []
    prompt1 = "Programming in C++ is like"
    inputs0.append(grpcclient.InferInput("text_input", [1, 1], "BYTES"))
    inputs0[0].set_data_from_numpy(np.array([[prompt1]], dtype=np.object_))

    prompt2 = "Programming in Assembly is like"
    inputs1 = []
    inputs1.append(grpcclient.InferInput("text_input", [1, 1], "BYTES"))
    inputs1[0].set_data_from_numpy(np.array([[prompt2]], dtype=np.object_))

    event1 = threading.Event()
    event2 = threading.Event()
    client1.start_stream(callback=partial(partial(client1_callback, display), event1))
    client2.start_stream(callback=partial(partial(client2_callback, display), event2))

    while True:

        event1.clear()
        event2.clear()

        display.clear()
        parameters = {"ignore_eos": True, "max_tokens": max_tokens}

        client1.async_stream_infer(
            model_name=model_name,
            inputs=inputs0,
            enable_empty_final_response=True,
            parameters=parameters,
        )

        time.sleep(0.05)
        client2.async_stream_infer(
            model_name=model_name,
            inputs=inputs1,
            enable_empty_final_response=True,
            parameters=parameters,
        )

        event1.wait()
        event2.wait()
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="localhost:8001")
    parser.add_argument("--model", type=str, default="simple-gpt2")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    display = Display(args.max_tokens)

    run_inferences(args.url, args.model, display, args.max_tokens)
