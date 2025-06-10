import argparse
import os
from glob import glob

import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
from PIL import Image

from tritony import InferenceClient


def preprocess(
    img,
    dtype=np.float32,
    format=mc.ModelInput.FORMAT_NCHW,
    c=3,
    h=224,
    w=224,
    scaling="INCEPTION",
):

    if c == 1:
        sample_img = img.convert("L")
    else:
        sample_img = img.convert("RGB")

    resized_img = sample_img.resize((w, h), Image.Resampling.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    resized = resized.astype(dtype)

    if scaling == "INCEPTION":
        scaled = (resized / 127.5) - 1
    elif scaling == "VGG":
        if c == 1:
            scaled = resized - np.asarray((128,))
        else:
            scaled = resized - np.asarray((123, 117, 104))

    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    return ordered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="Input folder.")
    FLAGS = parser.parse_args()

    client = InferenceClient.create_with(
        "densenet_onnx", "0.0.0.0:8001", input_dims=3, protocol="grpc"
    )
    client.output_kwargs = {"class_count": 1}

    image_data = []
    for filename in glob(os.path.join(FLAGS.image_folder, "*")):
        image_data.append(preprocess(Image.open(filename)))

    result = client(np.asarray(image_data))

    for output in result:
        max_value, arg_max, class_name = output[0].decode("utf-8").split(":")
        print(f"{max_value} ({arg_max}) = {class_name}")
