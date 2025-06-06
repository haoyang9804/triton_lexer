from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import BertTokenizerFast, TensorType


class TritonPythonModel:
    tokenizer: BertTokenizerFast

    def initialize(self, args: Dict[str, str]) -> None:

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":

        responses = []

        for request in requests:

            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
            tokens: Dict[str, np.ndarray] = self.tokenizer(
                text=query,
                return_tensors=TensorType.NUMPY,
                padding="max_length",
                max_length=256,
                truncation=True,
            )

            tokens = {k: v.astype(np.int64) for k, v in tokens.items()}

            outputs = list()
            for input_name in self.tokenizer.model_input_names:
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name])
                outputs.append(tensor_input)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")
