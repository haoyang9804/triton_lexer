import json
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])

        params = model_config["parameters"]

        accumulate_tokens_str = ""
        if "accumulate_tokens" in params:
            accumulate_tokens_str = params["accumulate_tokens"]["string_value"]

        self.accumulate_tokens = accumulate_tokens_str.lower() in [
            "true",
            "yes",
            "1",
            "t",
        ]

        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(model_config)

        self.logger = pb_utils.Logger

        self.bls_input_tensor_names = [
            "text_input",
            "max_tokens",
            "bad_words",
            "stop_words",
            "end_id",
            "pad_id",
            "top_k",
            "top_p",
            "temperature",
            "length_penalty",
            "repetition_penalty",
            "min_length",
            "presence_penalty",
            "frequency_penalty",
            "random_seed",
            "return_log_probs",
            "return_context_logits",
            "return_generation_logits",
            "beam_width",
            "stream",
            "prompt_embedding_table",
            "prompt_vocab_size",
            "embedding_bias_words",
            "embedding_bias_weights",
        ]

        self.preproc_input_to_bls_input_map = {
            "QUERY": "text_input",
            "REQUEST_OUTPUT_LEN": "max_tokens",
            "BAD_WORDS_DICT": "bad_words",
            "STOP_WORDS_DICT": "stop_words",
            "EMBEDDING_BIAS_WORDS": "embedding_bias_words",
            "EMBEDDING_BIAS_WEIGHTS": "embedding_bias_weights",
            "END_ID": "end_id",
            "PAD_ID": "pad_id",
        }

        self.preproc_output_to_trtllm_input_map = {
            "INPUT_ID": "input_ids",
            "REQUEST_INPUT_LEN": "input_lengths",
            "REQUEST_OUTPUT_LEN": "request_output_len",
            "BAD_WORDS_IDS": "bad_words_list",
            "STOP_WORDS_IDS": "stop_words_list",
            "EMBEDDING_BIAS": "embedding_bias",
            "OUT_END_ID": "end_id",
            "OUT_PAD_ID": "pad_id",
        }

        self.trtllm_input_to_bls_input_map = {
            "beam_width": "beam_width",
            "runtime_top_k": "top_k",
            "runtime_top_p": "top_p",
            "len_penalty": "length_penalty",
            "repetition_penalty": "repetition_penalty",
            "min_length": "min_length",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "random_seed": "random_seed",
            "return_log_probs": "return_log_probs",
            "return_context_logits": "return_context_logits",
            "return_generation_logits": "return_generation_logits",
            "streaming": "stream",
            "prompt_embedding_table": "prompt_embedding_table",
            "prompt_vocab_size": "prompt_vocab_size",
        }

        self.trtllm_output_to_postproc_input_map = {
            "output_ids": "TOKENS_BATCH",
            "sequence_length": "SEQUENCE_LENGTH",
            "cum_log_probs": "CUM_LOG_PROBS",
            "output_log_probs": "OUTPUT_LOG_PROBS",
            "context_logits": "CONTEXT_LOGITS",
            "generation_logits": "GENERATION_LOGITS",
        }

        self.postproc_output_to_bls_output_map = {
            "OUTPUT": "text_output",
            "OUT_CUM_LOG_PROBS": "cum_log_probs",
            "OUT_OUTPUT_LOG_PROBS": "output_log_probs",
            "OUT_CONTEXT_LOGITS": "context_logits",
            "OUT_GENERATION_LOGITS": "generation_logits",
        }

    def _get_bls_input_tensors_map(self, request):

        bls_input_tensors_map = {}
        for input_tensor_name in self.bls_input_tensor_names:
            tensor = pb_utils.get_input_tensor_by_name(request, input_tensor_name)
            if tensor != None:
                bls_input_tensors_map[input_tensor_name] = tensor

        return bls_input_tensors_map

    def _get_preproc_input_tensors(self, bls_input_tensors_map):

        preproc_input_tensors = []

        for preproc_name, bls_name in self.preproc_input_to_bls_input_map.items():

            if bls_name in bls_input_tensors_map:
                tensor = bls_input_tensors_map[bls_name]

                preproc_input_tensors.append(
                    pb_utils.Tensor(preproc_name, tensor.as_numpy())
                )

        return preproc_input_tensors

    def _get_trtllm_input_tensors(self, bls_input_tensors_map, preproc_output_tensors):

        trtllm_input_tensors = []

        for preproc_output_tensor in preproc_output_tensors:

            trtllm_tensor_name = self.preproc_output_to_trtllm_input_map[
                preproc_output_tensor.name()
            ]
            trtllm_input_tensors.append(
                pb_utils.Tensor(trtllm_tensor_name, preproc_output_tensor.as_numpy())
            )

        for trtllm_name, bls_name in self.trtllm_input_to_bls_input_map.items():

            if bls_name in bls_input_tensors_map:
                tensor = bls_input_tensors_map[bls_name]

                trtllm_input_tensors.append(
                    pb_utils.Tensor(trtllm_name, tensor.as_numpy())
                )

        return trtllm_input_tensors

    def _get_postproc_input_tensors(self, tokens, trtllm_output_tensors):

        postproc_input_tensors = []

        for trtllm_output_tensor in trtllm_output_tensors:

            if (
                self.accumulate_tokens
                and self.decoupled
                and trtllm_output_tensor.name() == "output_ids"
            ):

                new_tokens = trtllm_output_tensor.as_numpy()
                if new_tokens.ndim != 3:
                    raise pb_utils.TritonModelException(
                        "Expected output_ids tensor to have 3 dims."
                    )
                if new_tokens.shape[0] != 1:
                    raise pb_utils.TritonModelException(
                        "Expected output_ids tensor to have batch size of 1"
                    )
                if new_tokens.shape[1] != 1:
                    raise pb_utils.TritonModelException(
                        "Accumulation of tokens is only implemented for beam width = 1"
                    )

                tokens = (
                    new_tokens
                    if (tokens is None)
                    else np.concatenate((tokens, new_tokens), axis=2)
                )

                postproc_output_ids_name = self.trtllm_output_to_postproc_input_map[
                    "output_ids"
                ]
                postproc_input_tensors.append(
                    pb_utils.Tensor(postproc_output_ids_name, tokens)
                )

                np_seq_len_tensor = np.array([[tokens.shape[2]]], dtype=np.int32)
                postproc_seq_len_name = self.trtllm_output_to_postproc_input_map[
                    "sequence_length"
                ]
                postproc_input_tensors.append(
                    pb_utils.Tensor(postproc_seq_len_name, np_seq_len_tensor)
                )

        for trtllm_output_tensor in trtllm_output_tensors:

            if (
                self.accumulate_tokens
                and self.decoupled
                and (
                    trtllm_output_tensor.name() == "output_ids"
                    or trtllm_output_tensor.name() == "sequence_length"
                )
            ):
                continue

            postproc_tensor_name = self.trtllm_output_to_postproc_input_map[
                trtllm_output_tensor.name()
            ]

            postproc_input_tensors.append(
                pb_utils.Tensor(postproc_tensor_name, trtllm_output_tensor.as_numpy())
            )

        return tokens, postproc_input_tensors

    def _get_bls_output_tensors(self, postproc_output_tensors):

        bls_output_tensors = []

        for postproc_output_tensor in postproc_output_tensors:

            bls_tensor_name = self.postproc_output_to_bls_output_map[
                postproc_output_tensor.name()
            ]
            bls_output_tensors.append(
                pb_utils.Tensor(bls_tensor_name, postproc_output_tensor.as_numpy())
            )

        return bls_output_tensors

    def execute(self, requests):

        responses = []
        bls_response_sender = None

        for request in requests:

            if self.decoupled:
                bls_response_sender = request.get_response_sender()

            try:

                bls_input_tensors_map = self._get_bls_input_tensors_map(request)

                for name, tensor in bls_input_tensors_map.items():
                    batch_dim = tensor.as_numpy().shape[0]

                    if batch_dim != 1:

                        err_str = "Inflight batching backend expects requests with batch size of 1."
                        self.logger.log_error(err_str)
                        raise pb_utils.TritonModelException(err_str)

                preproc_input_tensors = self._get_preproc_input_tensors(
                    bls_input_tensors_map
                )

                preproc_request = pb_utils.InferenceRequest(
                    model_name="preprocessing",
                    inputs=preproc_input_tensors,
                    requested_output_names=list(
                        self.preproc_output_to_trtllm_input_map.keys()
                    ),
                )

                preproc_response = preproc_request.exec()

                if preproc_response.has_error():
                    raise pb_utils.TritonModelException(
                        preproc_response.error().message()
                    )

                trtllm_input_tensors = self._get_trtllm_input_tensors(
                    bls_input_tensors_map, preproc_response.output_tensors()
                )

                trtllm_request = pb_utils.InferenceRequest(
                    model_name="tensorrt_llm",
                    inputs=trtllm_input_tensors,
                    requested_output_names=list(
                        self.trtllm_output_to_postproc_input_map.keys()
                    ),
                )

                trtllm_responses = trtllm_request.exec(decoupled=self.decoupled)

                if not self.decoupled:
                    trtllm_responses = [trtllm_responses]

                tokens = None

                for trtllm_response in trtllm_responses:

                    if trtllm_response.has_error():
                        raise pb_utils.TritonModelException(
                            trtllm_response.error().message()
                        )

                    trtllm_output_tensors = trtllm_response.output_tensors()

                    tokens, postproc_input_tensors = self._get_postproc_input_tensors(
                        tokens, trtllm_output_tensors
                    )

                    postproc_request = pb_utils.InferenceRequest(
                        model_name="postprocessing",
                        inputs=postproc_input_tensors,
                        requested_output_names=list(
                            self.postproc_output_to_bls_output_map.keys()
                        ),
                    )

                    postproc_response = postproc_request.exec()

                    if postproc_response.has_error():
                        raise pb_utils.TritonModelException(
                            postproc_response.error().message()
                        )

                    bls_output_tensors = self._get_bls_output_tensors(
                        postproc_response.output_tensors()
                    )

                    bls_response = pb_utils.InferenceResponse(
                        output_tensors=bls_output_tensors
                    )

                    if self.decoupled:
                        bls_response_sender.send(bls_response)
                    else:
                        responses.append(bls_response)

                if self.decoupled:
                    bls_response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )

            except Exception:

                self.logger.log_error(traceback.format_exc())

                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(traceback.format_exc()),
                )

                if self.decoupled:
                    bls_response_sender.send(error_response)
                    bls_response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    responses.append(error_response)

        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses

    def finalize(self):

        print("Cleaning up...")
