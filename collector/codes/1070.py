import argparse
import json
import os
import sys

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from google.protobuf import json_format, text_format
from tritonclient.utils import *

FLAGS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected_dir",
        type=str,
        required=True,
        help="Directory containing expected output files",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    FLAGS, unparsed = parser.parse_known_args()

    for pair in [("localhost:8000", "http"), ("localhost:8001", "grpc")]:
        model_name = FLAGS.model
        if pair[1] == "http":
            triton_client = httpclient.InferenceServerClient(url=pair[0], verbose=False)
            model_config = triton_client.get_model_config(model_name)
        else:
            triton_client = grpcclient.InferenceServerClient(url=pair[0], verbose=False)
            model_config = triton_client.get_model_config(model_name)

        nonmatch = list()
        expected_files = [
            f
            for f in os.listdir(FLAGS.expected_dir)
            if (
                os.path.isfile(os.path.join(FLAGS.expected_dir, f))
                and (f.startswith("expected"))
            )
        ]
        for efile in expected_files:
            with open(os.path.join(FLAGS.expected_dir, efile)) as f:
                config = text_format.Parse(f.read(), mc.ModelConfig())

            if pair[1] == "http":
                config_json = json.loads(
                    json_format.MessageToJson(config, preserving_proto_field_name=True)
                )
                if config_json == model_config:
                    sys.exit(0)
            else:
                if config == model_config.config:
                    sys.exit(0)

        nonmatch.append(config)

    print("Model config doesn't match any expected output:")
    print("Model config:")
    print(model_config)
    for nm in nonmatch:
        print("Non-matching:")
        print(nm)

    sys.exit(1)
