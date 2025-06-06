import ctypes
import json
import os
import re
from dataclasses import asdict
from typing import Iterable, List, Optional, Union

import numpy as np
import tritonserver
from pydantic import BaseModel
from schemas.openai import (
    ChatCompletionNamedToolChoice,
    ChatCompletionToolChoiceOption1,
    CreateChatCompletionRequest,
    CreateCompletionRequest,
)


def _create_vllm_inference_request(
    model,
    prompt,
    request: CreateChatCompletionRequest | CreateCompletionRequest,
    lora_name: str | None,
):
    inputs = {}

    excludes = {
        "model",
        "stream",
        "messages",
        "prompt",
        "echo",
        "store",
        "metadata",
        "response_format",
        "service_tier",
        "stream_options",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "function_call",
        "functions",
        "suffix",
    }

    sampling_parameters = request.model_dump(
        exclude=excludes,
        exclude_none=True,
    )
    if lora_name is not None:
        sampling_parameters["lora_name"] = lora_name
    sampling_parameters = json.dumps(sampling_parameters)

    guided_json = _get_guided_json_from_tool(request)
    if guided_json is not None:
        from vllm.sampling_params import GuidedDecodingParams

        sampling_parameters_json = json.loads(sampling_parameters)
        sampling_parameters_json["guided_decoding"] = json.dumps(
            asdict(GuidedDecodingParams.from_optional(json=guided_json))
        )
        sampling_parameters = json.dumps(sampling_parameters_json)

    exclude_input_in_output = True
    echo = getattr(request, "echo", None)
    if echo is not None:
        exclude_input_in_output = not echo

    inputs["text_input"] = [prompt]
    inputs["stream"] = np.bool_([request.stream])
    inputs["exclude_input_in_output"] = np.bool_([exclude_input_in_output])

    inputs["sampling_parameters"] = [sampling_parameters]
    return model.create_request(inputs=inputs)


def _create_trtllm_inference_request(
    model,
    prompt,
    request: CreateChatCompletionRequest | CreateCompletionRequest,
    lora_name: str | None,
):
    if lora_name is not None:
        raise Exception("LoRA selection is currently not supported for TRT-LLM backend")

    inputs = {}
    inputs["text_input"] = [[prompt]]
    inputs["stream"] = np.bool_([[request.stream]])
    if request.max_tokens:
        inputs["max_tokens"] = np.int32([[request.max_tokens]])
    if request.stop:
        if isinstance(request.stop, str):
            request.stop = [request.stop]
        inputs["stop_words"] = [request.stop]

    if request.top_p is not None:
        inputs["top_p"] = np.float32([[request.top_p]])
    if request.frequency_penalty is not None:
        inputs["frequency_penalty"] = np.float32([[request.frequency_penalty]])
    if request.presence_penalty is not None:
        inputs["presence_penalty"] = np.float32([[request.presence_penalty]])
    if request.seed is not None:
        inputs["random_seed"] = np.uint64([[request.seed]])
    if request.temperature is not None:
        inputs["temperature"] = np.float32([[request.temperature]])

    guided_json = _get_guided_json_from_tool(request)
    if guided_json is not None:
        inputs["guided_decoding_guide_type"] = [["json_schema"]]
        inputs["guided_decoding_guide"] = [[guided_json]]

    return model.create_request(inputs=inputs)


def _construct_string_from_pointer(pointer: int, size: int) -> str:

    string_buffer = ctypes.create_string_buffer(size + 1)

    ctypes.memmove(string_buffer, pointer, size)

    return string_buffer.value.decode("utf-8")


def _get_volume(shape: Iterable[int]) -> int:
    volume = 1
    for dim in shape:
        volume *= dim

    return volume


def _to_string(tensor: tritonserver.Tensor) -> str:

    volume = _get_volume(tensor.shape)
    if volume != 1:
        raise Exception(
            f"Expected to find 1 string in the output, found {volume} instead."
        )
    if tensor.size < 4:
        raise Exception(
            f"Expected string buffer to contain its serialized byte size, but found size of {tensor.size}."
        )

    return _construct_string_from_pointer(tensor.data_ptr + 4, tensor.size - 4)


def _get_output(response: tritonserver._api._response.InferenceResponse) -> str:
    if "text_output" in response.outputs:
        tensor = response.outputs["text_output"]

        return _to_string(tensor)

    return ""


def _validate_triton_responses_non_streaming(
    responses: List[tritonserver._api._response.InferenceResponse],
):
    num_responses = len(responses)
    if num_responses == 1 and responses[0].final != True:
        raise Exception("Unexpected internal error with incorrect response flags")
    if num_responses == 2 and responses[-1].final != True:
        raise Exception("Unexpected internal error with incorrect response flags")
    if num_responses > 2:
        raise Exception(f"Unexpected number of responses: {num_responses}, expected 1.")


def _get_guided_json_from_tool(
    request: CreateChatCompletionRequest | CreateCompletionRequest,
) -> Optional[Union[str, dict, BaseModel]]:
    if isinstance(request, CreateChatCompletionRequest):
        if request.tool_choice is None or not request.tools:
            return None

        if type(request.tool_choice.root) is ChatCompletionNamedToolChoice:
            tool_name = request.tool_choice.root.function.name
        elif request.tool_choice.root == ChatCompletionToolChoiceOption1.required:
            tool_name = request.tools[0].function.name
        else:
            return None

        tools = {tool.function.name: tool.function for tool in request.tools}
        if tool_name not in tools:
            raise ValueError(f"Tool '{tool_name}' has not been passed in `tools`.")
        tool = tools[tool_name]
        return tool.parameters.model_dump_json()

    return None


def _get_vllm_lora_names(
    model_repository: str | list[str], model_name: str, model_version: int
) -> None | List[str]:
    lora_names = []
    repo_paths = model_repository
    if isinstance(repo_paths, str):
        repo_paths = [repo_paths]
    for repo_path in repo_paths:
        model_path = os.path.join(repo_path, model_name)
        if not os.path.isdir(model_path):

            return None
        if model_version <= 0:
            for version_path in os.listdir(model_path):
                version = os.path.basename(version_path)
                if re.fullmatch(r"^[0-9]+$", version) is None:
                    continue
                model_version = max(model_version, int(version))
            if model_version <= 0:

                return None
        version_path = os.path.join(model_path, str(model_version))
        is_lora_enabled = False
        model_file_path = os.path.join(version_path, "model.json")
        try:
            with open(model_file_path, "r") as f:
                config = json.load(f)
                if "enable_lora" in config:

                    is_lora_enabled = str(config["enable_lora"]).lower() == "true"
        except Exception:

            return None
        if is_lora_enabled != True:
            continue
        lora_config_path = os.path.join(version_path, "multi_lora.json")
        try:
            with open(lora_config_path, "r") as f:
                lora_config = json.load(f)
                for lora_name in lora_config.keys():
                    lora_names.append(lora_name)
        except Exception:

            return None
    return lora_names
