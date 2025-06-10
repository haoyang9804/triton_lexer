import os

import numpy as np
import requests as rq
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        HF_LOCATION = os.getenv("HF_LOCATION", pb_utils.get_model_dir())
        self.image_processor = AutoProcessor.from_pretrained(HF_LOCATION)
        self.logger = pb_utils.Logger
        self.tokenizer = AutoTokenizer.from_pretrained(HF_LOCATION)
        self.vocab_size = 32064
        self.max_input_len = 2048

    def _tokenize(self, prompt, num_visual_tokens):
        chunks = prompt.split("<image>")
        assert len(chunks) == 2, "Only support exactly one image per prompt"

        return (
            self.tokenizer.encode(chunks[0])
            + list(range(self.vocab_size, self.vocab_size + num_visual_tokens))
            + self.tokenizer.encode(chunks[1])[self.tokenizer.add_bos_token :]
        )

    def _parse_input(self, request, input_name, default=None):
        input = pb_utils.get_input_tensor_by_name(request, input_name)
        if input is not None:
            return input.as_numpy()[0]

        return default

    def _extract_image_features(self, image, prompt):

        pil_image = Image.open(rq.get(image, stream=True).raw).convert("RGB")
        image = self.image_processor(
            text=prompt, images=pil_image, return_tensors="np"
        )["pixel_values"].astype(np.float16)

        infer_request = pb_utils.InferenceRequest(
            model_name="vision_encoder",
            requested_output_names=["features"],
            inputs=[pb_utils.Tensor("image", image)],
        )
        vision_response = infer_request.exec()
        image_features = pb_utils.get_output_tensor_by_name(vision_response, "features")
        return torch.from_dlpack(image_features.as_numpy())

    def _prepare_llm_inputs(self, request, image_features, prompt):

        input_ids = self._tokenize(prompt, len(image_features[0]))
        input_ids = np.array(input_ids, dtype=np.int32)
        input_len = input_ids.shape[0]
        if input_len > self.max_input_len:
            return pb_utils.TritonError(
                f"Input length ({input_len:d}) exceeds limit ({self.max_input_len:d})"
            )
        max_tokens = self._parse_input(request, "max_tokens", default=50)
        temperature = self._parse_input(request, "temperature", default=0.5)
        top_k = self._parse_input(request, "top_k", default=1)
        frequency_penalty = self._parse_input(request, "frequency_penalty", default=0.7)
        seed = self._parse_input(request, "seed", default=10)
        embedding_args = {
            "prompt_vocab_size": np.array(
                [[image_features[0].shape[0]]], dtype=np.uint32
            ),
            "prompt_embedding_table": np.expand_dims(image_features[0], 0).astype(
                np.float16
            ),
        }

        return {
            "input_ids": np.expand_dims(input_ids, 0),
            "input_lengths": np.array([[input_len]], dtype=np.int32),
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "temperature": np.array([[temperature]], dtype=np.float32),
            "runtime_top_k": np.array([[top_k]], dtype=np.int32),
            "frequency_penalty": np.array([[frequency_penalty]], dtype=np.float32),
            "end_id": np.array([[self.tokenizer.eos_token_id]], dtype=np.int32),
            "random_seed": np.array([[seed]], dtype=np.uint64),
            "streaming": np.array([[1]], dtype=np.bool_),
            **embedding_args,
        }

    def _prepare_llm_response(self, llm_request_inputs):

        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=[pb_utils.Tensor(k, v) for k, v in llm_request_inputs.items()],
        )
        output_ids, output_len = [], 0
        max_len = llm_request_inputs["request_output_len"][0][0]

        for llm_response in llm_request.exec(decoupled=True):
            if llm_response.has_error():
                raise pb_utils.TritonModelException(llm_response.error().message())
            stream_output_ids = (
                pb_utils.get_output_tensor_by_name(llm_response, "output_ids")
                .as_numpy()
                .flatten()
                .tolist()
            )
            finish_reason = ""
            if len(stream_output_ids) == 0 or (
                len(stream_output_ids) != 0
                and stream_output_ids[-1] == self.tokenizer.eos_token_id
            ):
                finish_reason = "stop"
            output_ids += stream_output_ids
            if len(output_ids) >= max_len:
                finish_reason = "length"
                output_ids = output_ids[:max_len]
            last_response = finish_reason != ""
            output_len = len(output_ids)
            if last_response:
                output_text = self.tokenizer.decode(output_ids).strip()
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("text", np.array([output_text], np.object_)),
                        pb_utils.Tensor(
                            "finish_reason", np.array([finish_reason], np.object_)
                        ),
                        pb_utils.Tensor(
                            "completion_tokens", np.array([output_len], np.int32)
                        ),
                        pb_utils.Tensor(
                            "prompt_tokens",
                            np.array([llm_request_inputs["input_lengths"]], np.int32),
                        ),
                        pb_utils.Tensor(
                            "total_tokens",
                            np.array(
                                [output_len + llm_request_inputs["input_lengths"]],
                                np.int32,
                            ),
                        ),
                    ]
                )
                return response
        return None

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            image = (
                pb_utils.get_input_tensor_by_name(request, "image")
                .as_numpy()
                .flatten()
                .tolist()
            )
            if isinstance(image[0], bytes):
                image = image[0].decode("utf-8")

            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

            image_features = self._extract_image_features(image, prompt)

            llm_request_inputs = self._prepare_llm_inputs(
                request, image_features, prompt
            )
            if isinstance(llm_request_inputs, pb_utils.TritonError):
                error = pb_utils.InferenceResponse(error=llm_request_inputs)
                response_sender.send(
                    error, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                return

            llm_response = self._prepare_llm_response(llm_request_inputs)
            if llm_response is not None:
                response_sender.send(
                    llm_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

        return None
