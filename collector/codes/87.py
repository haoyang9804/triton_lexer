import json
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer


class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])

        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]

        tokenizer_type = model_config["parameters"]["tokenizer_type"]["string_value"]

        self.add_special_tokens = model_config["parameters"].get(
            "add_special_tokens", {"string_value": "false"}
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

        if self.tokenizer.pad_token_id is None:
            self.tokenizer_pad_id = self.tokenizer.eos_token_id
        self.tokenizer_pad_id = self.tokenizer.pad_token_id
        self.tokenizer_end_id = self.tokenizer.eos_token_id

        output_names = [
            "INPUT_ID",
            "REQUEST_INPUT_LEN",
            "BAD_WORDS_IDS",
            "STOP_WORDS_IDS",
            "OUT_END_ID",
            "OUT_PAD_ID",
        ]
        input_names = ["EMBEDDING_BIAS_WORDS", "EMBEDDING_BIAS_WEIGHTS"]
        for input_name in input_names:
            setattr(
                self,
                input_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_input_config_by_name(model_config, input_name)[
                        "data_type"
                    ]
                ),
            )

        for output_name in output_names:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(model_config, output_name)[
                        "data_type"
                    ]
                ),
            )

    def execute(self, requests):

        responses = []

        logger = pb_utils.Logger
        for idx, request in enumerate(requests):

            query = pb_utils.get_input_tensor_by_name(request, "QUERY").as_numpy()
            batch_dim = query.shape[0]
            if batch_dim != 1:

                err_str = (
                    "Inflight batching backend expects requests with batch size of 1."
                )
                logger.log_error(err_str)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(err_str)
                    )
                )
                continue

            request_output_len = pb_utils.get_input_tensor_by_name(
                request, "REQUEST_OUTPUT_LEN"
            ).as_numpy()

            bad_words_dict = pb_utils.get_input_tensor_by_name(
                request, "BAD_WORDS_DICT"
            )
            if bad_words_dict is not None:
                bad_words_dict = bad_words_dict.as_numpy()

            stop_words_dict = pb_utils.get_input_tensor_by_name(
                request, "STOP_WORDS_DICT"
            )
            if stop_words_dict is not None:
                stop_words_dict = stop_words_dict.as_numpy()

            embedding_bias_words = pb_utils.get_input_tensor_by_name(
                request, "EMBEDDING_BIAS_WORDS"
            )
            if embedding_bias_words is not None:
                embedding_bias_words = embedding_bias_words.as_numpy()

            embedding_bias_weights = pb_utils.get_input_tensor_by_name(
                request, "EMBEDDING_BIAS_WEIGHTS"
            )
            if embedding_bias_weights is not None:
                embedding_bias_weights = embedding_bias_weights.as_numpy()

            end_id = pb_utils.get_input_tensor_by_name(request, "END_ID")
            if end_id is not None:
                end_id = end_id.as_numpy()
            else:
                end_id = [[self.tokenizer_end_id]]

            pad_id = pb_utils.get_input_tensor_by_name(request, "PAD_ID")
            if pad_id is not None:
                pad_id = pad_id.as_numpy()
            else:
                pad_id = [[self.tokenizer_pad_id]]

            input_id, request_input_len = self._create_request(query)
            bad_words = self._to_word_list_format(bad_words_dict)
            stop_words = self._to_word_list_format(stop_words_dict)

            embedding_bias = self._get_embedding_bias(
                embedding_bias_words,
                embedding_bias_weights,
                self.embedding_bias_weights_dtype,
            )

            input_id_tensor = pb_utils.Tensor(
                "INPUT_ID", input_id.astype(self.input_id_dtype)
            )
            request_input_len_tensor = pb_utils.Tensor(
                "REQUEST_INPUT_LEN",
                request_input_len.astype(self.request_input_len_dtype),
            )
            request_output_len_tensor = pb_utils.Tensor(
                "REQUEST_OUTPUT_LEN", request_output_len
            )
            bad_words_ids_tensor = pb_utils.Tensor("BAD_WORDS_IDS", bad_words)
            stop_words_ids_tensor = pb_utils.Tensor("STOP_WORDS_IDS", stop_words)
            embedding_bias_tensor = pb_utils.Tensor("EMBEDDING_BIAS", embedding_bias)
            end_id_tensor = pb_utils.Tensor(
                "OUT_END_ID", np.array(end_id, dtype=np.int32)
            )
            pad_id_tensor = pb_utils.Tensor(
                "OUT_PAD_ID", np.array(pad_id, dtype=np.int32)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    input_id_tensor,
                    bad_words_ids_tensor,
                    stop_words_ids_tensor,
                    request_input_len_tensor,
                    request_output_len_tensor,
                    embedding_bias_tensor,
                    end_id_tensor,
                    pad_id_tensor,
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")

    def _create_request(self, query):

        start_ids = [
            np.array(
                self.tokenizer.build_chat_input(s[0].decode(), role="user")[
                    "input_ids"
                ][0]
            ).astype(int)
            for s in query
        ]
        start_lengths = np.array([[len(ids)] for ids in start_ids]).astype(int)

        max_len = 0
        for seq in start_ids:
            max_len = max(max_len, seq.shape[0])
        start_ids = np.stack(
            [
                np.pad(
                    seq,
                    (0, max_len - seq.shape[0]),
                    "constant",
                    constant_values=(0, self.tokenizer_pad_id),
                )
                for seq in start_ids
            ]
        )

        return start_ids, start_lengths

    def _to_word_list_format(self, word_lists: List[List[str | bytes]]):

        assert self.tokenizer != None, "need to set tokenizer"

        if word_lists is None:

            return np.empty([1, 2, 0], dtype="int32")

        flat_ids = []
        offsets = []
        for word_list in word_lists:
            item_flat_ids = []
            item_offsets = []

            for word in word_list:
                if isinstance(word, bytes):
                    word = word.decode()

                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))

    def _get_embedding_bias(
        self, embedding_bias_words, embedding_bias_weights, bias_dtype
    ):

        assert self.tokenizer != None, "need to set tokenizer"

        if embedding_bias_words is None or embedding_bias_weights is None:
            return np.empty([1, 0], dtype=self.embedding_bias_weights_dtype)

        batch_embedding_bias = []
        for words, weights in zip(embedding_bias_words, embedding_bias_weights):

            vocab_size = self.tokenizer.vocab_size
            embedding_bias = [0.0] * vocab_size

            assert len(words) == len(
                weights
            ), "Embedding bias words must have same dimension as embedding bias weights"

            for word, weight in zip(words, weights):
                if isinstance(word, bytes):
                    word = word.decode()
                ids = self.tokenizer.encode(word)

                if len(ids) == 0:
                    continue

                for id in ids:
                    embedding_bias[id] += weight

            batch_embedding_bias.append(np.array(embedding_bias))

        return np.array(batch_embedding_bias, dtype=bias_dtype)
