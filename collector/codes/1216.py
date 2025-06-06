import json

import numpy as np


import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "recognition_postprocessing_output"
        )

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):

        output0_dtype = self.output0_dtype

        responses = []

        def decodeText(scores):
            text = ""
            alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
            for i in range(scores.shape[0]):
                c = np.argmax(scores[i])
                if c != 0:
                    text += alphabet[c - 1]
                else:
                    text += "-"

            char_list = []
            for i in range(len(text)):
                if text[i] != "-" and (not (i > 0 and text[i] == text[i - 1])):
                    char_list.append(text[i])
            return "".join(char_list)

        for request in requests:

            in_1 = pb_utils.get_input_tensor_by_name(
                request, "recognition_postprocessing_input"
            ).as_numpy()
            text_list = []
            for i in range(in_1.shape[0]):
                text_list.append(decodeText(in_1[i]))
            print(text_list, flush=True)
            out_tensor_0 = pb_utils.Tensor(
                "recognition_postprocessing_output",
                np.array(text_list).astype(output0_dtype),
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")
