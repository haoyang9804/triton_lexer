import io
import json
import logging
import os

import numpy as np


import triton_python_backend_utils as pb_utils
from PIL import Image
from pytesseract import apply_tesseract
from transformers import LayoutLMv3Processor

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TritonPythonModel:

    def initialize(self, args):

        self.model_config = model_config = json.loads(args["model_config"])
        self.preprocessor = LayoutLMv3Processor.from_pretrained(
            "{}/preprocessing_config".format(
                os.path.realpath(os.path.dirname(__file__))
            ),
            apply_ocr=False,
        )

        output0_config = pb_utils.get_output_config_by_name(model_config, "input_ids")

        output1_config = pb_utils.get_output_config_by_name(
            model_config, "attention_mask"
        )

        output2_config = pb_utils.get_output_config_by_name(
            model_config, "offset_mapping"
        )

        output3_config = pb_utils.get_output_config_by_name(model_config, "bbox")

        output4_config = pb_utils.get_output_config_by_name(
            model_config, "pixel_values"
        )

        output5_config = pb_utils.get_output_config_by_name(
            model_config, "raw_image_shape"
        )

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        self.output2_dtype = pb_utils.triton_string_to_numpy(
            output2_config["data_type"]
        )

        self.output3_dtype = pb_utils.triton_string_to_numpy(
            output3_config["data_type"]
        )

        self.output4_dtype = pb_utils.triton_string_to_numpy(
            output4_config["data_type"]
        )

        self.output5_dtype = pb_utils.triton_string_to_numpy(
            output5_config["data_type"]
        )

    def execute(self, requests):

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype
        output3_dtype = self.output3_dtype
        output4_dtype = self.output4_dtype
        output5_dtype = self.output5_dtype

        responses = []

        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(request, "raw_image_array")
            img = in_0.as_numpy()
            image = Image.open(io.BytesIO(img.tobytes())).convert("RGB")
            text, boxes = apply_tesseract(image, lang="eng", tesseract_config="--oem 1")
            h, w = image.size

            encoding = self.preprocessor(
                image, text=text, boxes=boxes, return_offsets_mapping=True
            )

            ipnut_ids = np.expand_dims(
                np.array(encoding["input_ids"]).astype(output0_dtype), axis=0
            )
            out_tensor_0 = pb_utils.Tensor("input_ids", ipnut_ids)

            attention_mask = np.expand_dims(
                np.array(encoding["attention_mask"]).astype(output1_dtype),
                axis=0,
            )
            out_tensor_1 = pb_utils.Tensor("attention_mask", attention_mask)

            offset_mapping = np.expand_dims(
                np.array(encoding["offset_mapping"]).astype(output2_dtype),
                axis=0,
            )
            out_tensor_2 = pb_utils.Tensor("offset_mapping", offset_mapping)

            bbox = np.expand_dims(
                np.array(encoding["bbox"]).astype(output3_dtype), axis=0
            )
            out_tensor_3 = pb_utils.Tensor("bbox", bbox)

            pixel_values = np.array(encoding["pixel_values"]).astype(output4_dtype)
            out_tensor_4 = pb_utils.Tensor("pixel_values", pixel_values)

            raw_image_shape = np.expand_dims(
                np.array((h, w)).astype(output5_dtype), axis=0
            )
            out_tensor_5 = pb_utils.Tensor("raw_image_shape", raw_image_shape)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    out_tensor_0,
                    out_tensor_1,
                    out_tensor_2,
                    out_tensor_3,
                    out_tensor_4,
                    out_tensor_5,
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Preprocessing cleaning up...")
