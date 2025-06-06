import copy
import subprocess
from pathlib import Path
from typing import List

import pytest
import tritonserver
from fastapi.testclient import TestClient
from tests.utils import setup_fastapi_app, setup_server


@pytest.mark.fastapi
class TestChatCompletions:
    @pytest.fixture(scope="class")
    def client(self, fastapi_client_class_scope):
        yield fastapi_client_class_scope

    def test_chat_completions_defaults(self, client, model: str, messages: List[dict]):
        response = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages},
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

        assert not response.json()["usage"]

    def test_chat_completions_system_prompt(self, client, model: str):

        messages = [
            {"role": "system", "content": "You are a Triton Inference Server expert."},
            {"role": "user", "content": "What is machine learning?"},
        ]

        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    def test_chat_completions_system_prompt_only(self, client, model: str):

        messages = [
            {"role": "system", "content": "You are a Triton Inference Server expert."}
        ]

        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    def test_chat_completions_user_prompt_str(self, client, model: str):

        messages = [{"role": "user", "content": "What is machine learning?"}]

        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    def test_chat_completions_user_prompt_dict(self, client, model: str):

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is machine learning?"}],
            }
        ]

        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    @pytest.mark.parametrize(
        "param_key, param_value",
        [
            ("temperature", 0.7),
            ("max_tokens", 10),
            ("top_p", 0.9),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.2),
            ("n", 1),
            ("stop", "."),
            ("stop", []),
            ("stop", [".", ","]),
            ("logprobs", True),
            ("logit_bias", {"0": 0}),
            ("min_tokens", 16),
            ("ignore_eos", True),
        ],
    )
    def test_chat_completions_sampling_parameters(
        self, client, param_key, param_value, model: str, messages: List[dict]
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                param_key: param_value,
            },
        )

        unsupported_parameters = ["logprobs", "logit_bias"]
        if param_key in unsupported_parameters:
            assert response.status_code == 400
            assert (
                response.json()["detail"]
                == "logit bias and log probs not currently supported"
            )
            return

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"]
        assert response.json()["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.parametrize(
        "param_key, param_value",
        [
            ("temperature", 2.1),
            ("temperature", -0.1),
            ("max_tokens", -1),
            ("top_p", 1.1),
            ("frequency_penalty", 3),
            ("frequency_penalty", -3),
            ("presence_penalty", 2.1),
            ("presence_penalty", -2.1),
            ("min_tokens", -1),
            ("ignore_eos", 123),
        ],
    )
    def test_chat_completions_invalid_sampling_parameters(
        self, client, param_key, param_value, model: str, messages: List[dict]
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                param_key: param_value,
            },
        )
        print("Response:", response.json())

        assert response.status_code == 422

    def test_chat_completions_max_tokens(
        self, client, model: str, messages: List[dict]
    ):
        responses = []
        payload = {"model": model, "messages": messages, "max_tokens": 1}

        payload["max_tokens"] = 1
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )

        payload["max_tokens"] = 100
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = (
            responses[0].json()["choices"][0]["message"]["content"].strip().split()
        )
        response2_text = (
            responses[1].json()["choices"][0]["message"]["content"].strip().split()
        )
        response3_text = (
            responses[2].json()["choices"][0]["message"]["content"].strip().split()
        )

        assert len(response1_text) == len(response2_text) == 1
        assert len(response3_text) > len(response1_text)

    @pytest.mark.parametrize(
        "temperature",
        [0.0, 1.0],
    )
    def test_chat_completions_temperature_vllm(
        self, client, temperature, backend: str, model: str, messages: List[dict]
    ):
        if backend != "vllm":
            pytest.skip(reason="Only used to test vLLM-specific temperature behavior")

        responses = []
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 256,
            "temperature": temperature,
        }

        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = (
            responses[0].json()["choices"][0]["message"]["content"].strip().split()
        )
        response2_text = (
            responses[1].json()["choices"][0]["message"]["content"].strip().split()
        )

        if temperature == 0.0:

            assert response1_text == response2_text

        elif temperature == 1.0:
            assert response1_text != response2_text

        else:
            raise ValueError(f"Unexpected {temperature=} for this test.")

    @pytest.mark.xfail(
        reason="TRT-LLM BLS model will ignore temperature until a later release"
    )
    def test_chat_completions_temperature_tensorrtllm(
        self, client, backend: str, model: str, messages: List[dict]
    ):
        if backend != "tensorrtllm":
            pytest.skip(
                reason="Only used to test TRT-LLM-specific temperature behavior"
            )

        responses = []
        payload1 = {
            "model": model,
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.0,
            "top_p": 0.5,
        }

        payload2 = copy.deepcopy(payload1)
        payload2["temperature"] = 1.0

        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload1,
            )
        )
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload1,
            )
        )

        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload2,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = (
            responses[0].json()["choices"][0]["message"]["content"].strip().split()
        )
        response2_text = (
            responses[1].json()["choices"][0]["message"]["content"].strip().split()
        )
        response3_text = (
            responses[2].json()["choices"][0]["message"]["content"].strip().split()
        )

        assert response1_text == response2_text
        assert response1_text != response3_text

    def test_chat_completions_seed(self, client, model: str, messages: List[dict]):
        responses = []
        payload1 = {
            "model": model,
            "messages": messages,
            "max_tokens": 200,
            "seed": 1,
        }
        payload2 = copy.deepcopy(payload1)
        payload2["seed"] = 2

        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload1,
            )
        )
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload1,
            )
        )

        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload2,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = (
            responses[0].json()["choices"][0]["message"]["content"].strip().split()
        )
        response2_text = (
            responses[1].json()["choices"][0]["message"]["content"].strip().split()
        )
        response3_text = (
            responses[2].json()["choices"][0]["message"]["content"].strip().split()
        )

        assert response1_text == response2_text
        assert response1_text != response3_text

    def test_chat_completions_no_message(
        self, client, model: str, messages: List[dict]
    ):

        messages = []
        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )
        assert response.status_code == 422
        assert (
            response.json()["detail"][0]["msg"]
            == "List should have at least 1 item after validation, not 0"
        )

    def test_chat_completions_empty_message(
        self, client, model: str, messages: List[dict]
    ):

        messages = [{}]
        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Field required"

    def test_chat_completions_multiple_choices(
        self, client, model: str, messages: List[dict]
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages, "n": 2},
        )

        assert response.status_code == 400
        assert "only single choice" in response.json()["detail"]

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_chat_completions_streaming(self, client):
        pass

    def test_chat_completions_no_streaming(
        self, client, model: str, messages: List[dict]
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages, "stream": False},
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_function_calling(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_lora(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_multi_lora(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_request_n_choices(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_request_logprobs(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_request_logit_bias(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_usage_response(self):
        pass


@pytest.mark.fastapi
class TestChatCompletionsTokenizers:

    @pytest.fixture(scope="class")
    def server(self, model_repository: str):
        server = setup_server(model_repository)
        yield server
        server.stop()

    def test_chat_completions_no_tokenizer(
        self,
        server: tritonserver.Server,
        backend: str,
        model: str,
        messages: List[dict],
    ):
        app = setup_fastapi_app(tokenizer="", server=server, backend=backend)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": messages},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "Unknown tokenizer"

    def test_chat_completions_custom_tokenizer(
        self,
        server: tritonserver.Server,
        backend: str,
        tokenizer_model: str,
        model: str,
        messages: List[dict],
    ):

        custom_tokenizer_path = str(Path(__file__).parent / "custom_tokenizer")
        download_cmd = f"huggingface-cli download --local-dir {custom_tokenizer_path} {tokenizer_model} --include *.json"
        print(f"Running download command: {download_cmd}")
        subprocess.run(download_cmd.split(), check=True)

        app_local = setup_fastapi_app(
            tokenizer=custom_tokenizer_path, server=server, backend=backend
        )
        app_hf = setup_fastapi_app(
            tokenizer=tokenizer_model, server=server, backend=backend
        )

        responses = []
        with TestClient(app_local) as client_local, TestClient(app_hf) as client_hf:
            payload = {"model": model, "messages": messages, "temperature": 0}
            responses.append(client_local.post("/v1/chat/completions", json=payload))
            responses.append(client_hf.post("/v1/chat/completions", json=payload))

        for response in responses:
            assert response.status_code == 200
            message = response.json()["choices"][0]["message"]
            assert message["content"].strip()
            assert message["role"] == "assistant"

        def equal_dicts(d1, d2, ignore_keys):
            d1_filtered = {k: v for k, v in d1.items() if k not in ignore_keys}
            d2_filtered = {k: v for k, v in d2.items() if k not in ignore_keys}
            return d1_filtered == d2_filtered

        ignore_keys = ["id", "created"]
        assert equal_dicts(
            responses[0].json(), responses[1].json(), ignore_keys=ignore_keys
        )

    def test_chat_completions_invalid_chat_tokenizer(
        self,
        server: tritonserver.Server,
        backend: str,
        model: str,
        messages: List[dict],
    ):

        import transformers

        print(f"{transformers.__version__=}")
        if transformers.__version__ < "4.44.0":
            pytest.xfail()

        invalid_chat_tokenizer = "gpt2"
        try:
            app = setup_fastapi_app(
                tokenizer=invalid_chat_tokenizer, server=server, backend=backend
            )
        except OSError as e:
            expected_msg = f"We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like {invalid_chat_tokenizer} is not the path to a directory containing a file named config.json."
            if expected_msg in str(e):
                pytest.skip("HuggingFace network issues")
            raise e
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": messages},
            )

        assert response.status_code == 400

        expected_errors = [
            "cannot use apply_chat_template()",
            "cannot use chat template",
        ]
        assert any(
            error in response.json()["detail"].lower() for error in expected_errors
        )
