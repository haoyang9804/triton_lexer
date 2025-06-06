import json

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer


class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])
        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]
        tokenizer_type = model_config["parameters"]["tokenizer_type"]["string_value"]
        self.skip_special_tokens = model_config["parameters"].get(
            "skip_special_tokens", {"string_value": "true"}
        )["string_value"].lower() in ["true", "1", "t", "y", "yes"]

        if tokenizer_type == "t5":
            self.tokenizer = T5Tokenizer(vocab_file=tokenizer_dir, padding_side="left")
        elif tokenizer_type == "auto":
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                legacy=False,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=True,
            )
        elif tokenizer_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_dir, legacy=False, padding_side="left"
            )
        else:
            raise AttributeError(f"Unexpected tokenizer type: {tokenizer_type}")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):

        responses = []

        for idx, request in enumerate(requests):

            tokens_batch = pb_utils.get_input_tensor_by_name(
                request, "TOKENS_BATCH"
            ).as_numpy()

            sequence_lengths = pb_utils.get_input_tensor_by_name(
                request, "SEQUENCE_LENGTH"
            ).as_numpy()

            cum_log_probs = pb_utils.get_input_tensor_by_name(
                request, "CUM_LOG_PROBS"
            ).as_numpy()

            output_log_probs = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_LOG_PROBS"
            ).as_numpy()

            context_logits = pb_utils.get_input_tensor_by_name(
                request, "CONTEXT_LOGITS"
            )

            if context_logits is not None:
                context_logits = context_logits.as_numpy()

            generation_logits = pb_utils.get_input_tensor_by_name(
                request, "GENERATION_LOGITS"
            )
            if generation_logits is not None:
                generation_logits = generation_logits.as_numpy()

            outputs = self._postprocessing(tokens_batch, sequence_lengths)

            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array(outputs).astype(self.output_dtype)
            )

            out_cum_log_probs = pb_utils.Tensor("OUT_CUM_LOG_PROBS", cum_log_probs)

            out_output_log_probs = pb_utils.Tensor(
                "OUT_OUTPUT_LOG_PROBS", output_log_probs
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor, out_cum_log_probs, out_output_log_probs]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")

    def _postprocessing(self, tokens_batch, sequence_lengths):
        outputs = []
        for batch_idx, beam_tokens in enumerate(tokens_batch):
            for beam_idx, tokens in enumerate(beam_tokens):
                seq_len = sequence_lengths[batch_idx][beam_idx]

                output = self.tokenizer.decode(tokens[:seq_len])
                print(output)
                outputs.append(output.encode("utf8"))
        return outputs
