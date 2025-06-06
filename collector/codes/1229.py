import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
import yfinance as yf
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import queue
import sys

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_stock_price",
            "description": "Get the current stock price for a given symbol.\n\nArgs:\n  symbol (str): The stock symbol.\n\nReturns:\n  float: The current stock price, or None if an error occurs.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Get company news and press releases for a given stock symbol.\n\nArgs:\nsymbol (str): The stock symbol.\n\nReturns:\npd.DataFrame: DataFrame containing company news and press releases.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return final generated answer",
            "parameters": {
                "type": "object",
                "properties": {"final_response": {"type": "string"}},
                "required": ["final_response"],
            },
        },
    },
]


class MyFunctions:
    def get_company_news(self, symbol: str) -> pd.DataFrame:

        try:
            news = yf.Ticker(symbol).news
            title_list = []
            for entry in news:
                title_list.append(entry["title"])
            return title_list
        except Exception as e:
            print(f"Error fetching company news for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_stock_price(self, symbol: str) -> float:

        try:
            stock = yf.Ticker(symbol)

            current_price = stock.info.get(
                "regularMarketPrice", stock.info.get("currentPrice")
            )
            return current_price if current_price else None
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None


class FunctionCall(BaseModel):
    step: str

    description: str

    tool: str

    arguments: dict


class PromptSchema(BaseModel):
    Role: str

    Objective: str

    Tools: str

    Schema: str

    Instructions: str


def read_yaml_file(file_path: str) -> PromptSchema:

    with open(file_path, "r") as file:
        yaml_content = yaml.safe_load(file)

    prompt_schema = PromptSchema(
        Role=yaml_content.get("Role", ""),
        Objective=yaml_content.get("Objective", ""),
        Tools=yaml_content.get("Tools", ""),
        Schema=yaml_content.get("Schema", ""),
        Instructions=yaml_content.get("Instructions", ""),
    )
    return prompt_schema


def format_yaml_prompt(prompt_schema: PromptSchema, variables: Dict) -> str:

    formatted_prompt = ""
    for field, value in prompt_schema.model_dump().items():
        formatted_value = value.format(**variables)
        if field == "Instructions":
            formatted_prompt += f"{formatted_value}"
        else:
            formatted_value = formatted_value.replace("\n", " ")
            formatted_prompt += f"{formatted_value}"
    return formatted_prompt


def process_prompt(
    user_prompt,
    system_prompt_yml=Path(__file__).parent.joinpath("./system_prompt_schema.yml"),
    tools=TOOLS,
    schema_json=FunctionCall.model_json_schema(),
):

    prompt_schema = read_yaml_file(system_prompt_yml)
    variables = {"tools": tools, "schema": schema_json}
    sys_prompt = format_yaml_prompt(prompt_schema, variables)
    processed_prompt = f"<|im_start|>system\n {sys_prompt}<|im_end|>\n"
    processed_prompt += f"<|im_start|>user\n {user_prompt}\nThis is the first turn and you don't have <tool_results> to analyze yet. <|im_end|>\n <|im_start|>assistant"
    return processed_prompt


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def run_inference(
    triton_client,
    prompt,
    output_len,
    request_id,
    repetition_penalty,
    presence_penalty,
    frequency_penalty,
    temperature,
    stop_words,
    bad_words,
    embedding_bias_words,
    embedding_bias_weights,
    model_name,
    streaming,
    beam_width,
    overwrite_output_text,
    return_context_logits_data,
    return_generation_logits_data,
    end_id,
    pad_id,
    verbose,
    num_draft_tokens=0,
    use_draft_logits=None,
):
    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.int32) * output_len
    streaming_data = np.array([[streaming]], dtype=bool)
    beam_width_data = np.array([[beam_width]], dtype=np.int32)
    temperature_data = np.array([[temperature]], dtype=np.float32)

    inputs = [
        prepare_tensor("text_input", input0_data),
        prepare_tensor("max_tokens", output0_len),
        prepare_tensor("stream", streaming_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
    ]

    if num_draft_tokens > 0:
        inputs.append(
            prepare_tensor(
                "num_draft_tokens", np.array([[num_draft_tokens]], dtype=np.int32)
            )
        )
    if use_draft_logits is not None:
        inputs.append(
            prepare_tensor(
                "use_draft_logits", np.array([[use_draft_logits]], dtype=bool)
            )
        )

    if bad_words:
        bad_words_list = np.array([bad_words], dtype=object)
        inputs += [prepare_tensor("bad_words", bad_words_list)]

    if stop_words:
        stop_words_list = np.array([stop_words], dtype=object)
        inputs += [prepare_tensor("stop_words", stop_words_list)]

    if repetition_penalty is not None:
        repetition_penalty = [[repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty, dtype=np.float32)
        inputs += [prepare_tensor("repetition_penalty", repetition_penalty_data)]

    if presence_penalty is not None:
        presence_penalty = [[presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)
        inputs += [prepare_tensor("presence_penalty", presence_penalty_data)]

    if frequency_penalty is not None:
        frequency_penalty = [[frequency_penalty]]
        frequency_penalty_data = np.array(frequency_penalty, dtype=np.float32)
        inputs += [prepare_tensor("frequency_penalty", frequency_penalty_data)]

    if return_context_logits_data is not None:
        inputs += [
            prepare_tensor("return_context_logits", return_context_logits_data),
        ]

    if return_generation_logits_data is not None:
        inputs += [
            prepare_tensor("return_generation_logits", return_generation_logits_data),
        ]

    if (embedding_bias_words is not None and embedding_bias_weights is None) or (
        embedding_bias_words is None and embedding_bias_weights is not None
    ):
        assert 0, "Both embedding bias words and weights must be specified"

    if embedding_bias_words is not None and embedding_bias_weights is not None:
        assert len(embedding_bias_words) == len(
            embedding_bias_weights
        ), "Embedding bias weights and words must have same length"
        embedding_bias_words_data = np.array([embedding_bias_words], dtype=object)
        embedding_bias_weights_data = np.array(
            [embedding_bias_weights], dtype=np.float32
        )
        inputs.append(prepare_tensor("embedding_bias_words", embedding_bias_words_data))
        inputs.append(
            prepare_tensor("embedding_bias_weights", embedding_bias_weights_data)
        )
    if end_id is not None:
        end_id_data = np.array([[end_id]], dtype=np.int32)
        inputs += [prepare_tensor("end_id", end_id_data)]

    if pad_id is not None:
        pad_id_data = np.array([[pad_id]], dtype=np.int32)
        inputs += [prepare_tensor("pad_id", pad_id_data)]

    user_data = UserData()

    triton_client.start_stream(callback=partial(callback, user_data))

    triton_client.async_stream_infer(model_name, inputs, request_id=request_id)

    triton_client.stop_stream()

    output_text = ""
    while True:
        try:
            result = user_data._completed_requests.get(block=False)
        except Exception:
            break

        if type(result) == InferenceServerException:
            print("Received an error from server:")
            print(result)
        else:
            output = result.as_numpy("text_output")
            if streaming and beam_width == 1:
                new_output = output[0].decode("utf-8")
                if overwrite_output_text:
                    output_text = new_output
                else:
                    output_text += new_output
            else:
                output_text = output[0].decode("utf-8")
                if verbose:
                    print(
                        str("\n[VERBOSE MODE] LLM's response:" + output_text),
                        flush=True,
                    )

            if return_context_logits_data is not None:
                context_logits = result.as_numpy("context_logits")
                if verbose:
                    print(f"context_logits.shape: {context_logits.shape}")
                    print(f"context_logits: {context_logits}")
            if return_generation_logits_data is not None:
                generation_logits = result.as_numpy("generation_logits")
                if verbose:
                    print(f"generation_logits.shape: {generation_logits.shape}")
                    print(f"generation_logits: {generation_logits}")

    if streaming and beam_width == 1:
        if verbose:
            print(output_text)

    return output_text
