import argparse
import sys

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

    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose
        )
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit(1)

    model_name = "simple"

    if len(triton_client.get_model_repository_index().models) != 8:
        print("FAILED : Repository Index")
        sys.exit(1)

    triton_client.load_model(model_name)
    if not triton_client.is_model_ready(model_name):
        print("FAILED : Load Model")
        sys.exit(1)

    try:
        config = '"parameters": {"config": {{"max_batch_size": "16"}}}'
        triton_client.load_model(model_name, config=config)
    except InferenceServerException as e:
        if "failed to load" not in e.message():
            sys.exit(1)
    else:
        print("Expect error occurs for invalid override config.")
        sys.exit(1)

    config = '{"max_batch_size":"16"}'
    triton_client.load_model(model_name, config=config)

    updated_model_config = triton_client.get_model_config(model_name)
    max_batch_size = updated_model_config.config.max_batch_size
    if max_batch_size != 16:
        print("Expect max_batch_size = 16, got: {}".format(max_batch_size))
        sys.exit(1)

    triton_client.unload_model(model_name)
    if triton_client.is_model_ready(model_name):
        print("FAILED : Unload Model")
        sys.exit(1)

    try:
        triton_client.load_model("wrong_model_name")
    except InferenceServerException as e:
        if "failed to load" in e.message():
            print("PASS: model control")
            sys.exit(0)

    print("FAILED")
    sys.exit(1)
