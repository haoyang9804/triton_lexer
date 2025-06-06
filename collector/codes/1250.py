import json
import logging

import numpy as np


import triton_python_backend_utils as pb_utils

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TritonPythonModel:

    def initialize(self, args):

        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "true_predictions"
        )

        output1_config = pb_utils.get_output_config_by_name(model_config, "true_boxes")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        self.id2label = {
            0: "O",
            1: "B-HEADER",
            2: "I-HEADER",
            3: "B-QUESTION",
            4: "I-QUESTION",
            5: "B-ANSWER",
            6: "I-ANSWER",
        }

    def unnormalize_box(self, bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]

    def iob_to_label(self, label):
        label = label[2:]
        if not label:
            return "other"
        return label

    def execute(self, requests):

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []

        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(request, "logits")
            in_1 = pb_utils.get_input_tensor_by_name(request, "bbox")
            in_2 = pb_utils.get_input_tensor_by_name(request, "offset_mapping")
            in_3 = pb_utils.get_input_tensor_by_name(request, "raw_image_shape")

            width, height = in_3.as_numpy().squeeze()
            token_boxes = in_1.as_numpy().squeeze().tolist()
            is_subword = np.array(in_2.as_numpy().squeeze().tolist())[:, 0] != 0

            predictions = in_0.as_numpy().argmax(-1).squeeze().tolist()

            true_predictions = [
                self.id2label[pred]
                for idx, pred in enumerate(predictions)
                if not is_subword[idx]
            ]

            true_boxes = [
                self.unnormalize_box(box, width, height)
                for idx, box in enumerate(token_boxes)
                if not is_subword[idx]
            ]

            out_tensor_0 = pb_utils.Tensor(
                "true_predictions", np.array(true_predictions).astype(output0_dtype)
            )

            out_tensor_1 = pb_utils.Tensor(
                "true_boxes", np.array(true_boxes).astype(output1_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Postprocess cleaning up...")
