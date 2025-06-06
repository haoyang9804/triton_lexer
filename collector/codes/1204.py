import json

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])
        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]

        skip_special_tokens = model_config["parameters"].get("skip_special_tokens")
        if skip_special_tokens is not None:
            skip_special_tokens_str = skip_special_tokens["string_value"].lower()
            if skip_special_tokens_str in [
                "true",
                "false",
                "1",
                "0",
                "t",
                "f",
                "y",
                "n",
                "yes",
                "no",
            ]:
                self.skip_special_tokens = skip_special_tokens_str in [
                    "true",
                    "1",
                    "t",
                    "y",
                    "yes",
                ]
            else:
                print(
                    f"[TensorRT-LLM][WARNING] Don't setup 'skip_special_tokens' correctly (set value is {skip_special_tokens['string_value']}). Set it as True by default."
                )
                self.skip_special_tokens = True
        else:
            print(
                f"[TensorRT-LLM][WARNING] Don't setup 'skip_special_tokens'. Set it as True by default."
            )
            self.skip_special_tokens = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", trust_remote_code=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):

        tokens_batch = []
        sequence_lengths = []
        for idx, request in enumerate(requests):
            for input_tensor in request.inputs():
                if input_tensor.name() == "TOKENS_BATCH":
                    tokens_batch.append(input_tensor.as_numpy())
                elif input_tensor.name() == "SEQUENCE_LENGTH":
                    sequence_lengths.append(input_tensor.as_numpy())
                else:
                    raise ValueError(f"unknown input {input_tensor.name}")

        list_of_tokens = []
        req_idx_offset = 0
        req_idx_offsets = [req_idx_offset]
        for idx, token_batch in enumerate(tokens_batch):
            for batch_idx, beam_tokens in enumerate(token_batch):
                for beam_idx, tokens in enumerate(beam_tokens):
                    seq_len = sequence_lengths[idx][batch_idx][beam_idx]
                    list_of_tokens.append(tokens[:seq_len])
                    req_idx_offset += 1

            req_idx_offsets.append(req_idx_offset)

        all_outputs = self.tokenizer.batch_decode(
            list_of_tokens, skip_special_tokens=self.skip_special_tokens
        )

        responses = []
        for idx, request in enumerate(requests):
            req_outputs = [
                x.encode("utf8")
                for x in all_outputs[req_idx_offsets[idx] : req_idx_offsets[idx + 1]]
            ]

            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array(req_outputs).astype(self.output_dtype)
            )

            outputs = [output_tensor]

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")
