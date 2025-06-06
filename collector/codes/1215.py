import io
import json

import numpy as np
import torch
import torchvision.transforms as transforms


import triton_python_backend_utils as pb_utils
from PIL import Image


class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "detection_preprocessing_output"
        )

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):

        output0_dtype = self.output0_dtype

        responses = []

        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(
                request, "detection_preprocessing_input"
            )

            def image_loader(image):
                [h, w] = image.size
                resize_w = (w // 32) * 32
                resize_h = (h // 32) * 32

                center_crop = resize_h if resize_w > resize_h else resize_w
                loader = transforms.Compose(
                    [transforms.CenterCrop(center_crop), transforms.ToTensor()]
                )

                im = loader(image)
                im = torch.unsqueeze(im, 0)
                return im.permute(0, 2, 3, 1)

            img = in_0.as_numpy()

            image = Image.open(io.BytesIO(img.tobytes()))
            img_out = image_loader(image)
            img_out = np.array(img_out) * 255.0

            out_tensor_0 = pb_utils.Tensor(
                "detection_preprocessing_output", img_out.astype(output0_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")
