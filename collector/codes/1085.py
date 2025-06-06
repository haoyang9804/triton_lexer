import os
import re
from functools import partial
from typing import Tuple, Union

import numpy as np
import requests
import tritonserver
from tritonfrontend import KServeGrpc, KServeHttp, Metrics


def setup_server(model_repository="test_model_repository") -> tritonserver.Server:

    module_directory = os.path.split(os.path.abspath(__file__))[0]
    model_path = os.path.abspath(os.path.join(module_directory, model_repository))

    server_options = tritonserver.Options(
        server_id="TestServer",
        model_repository=model_path,
        log_error=True,
        log_warn=True,
        log_info=True,
    )

    return tritonserver.Server(server_options).start(wait_until_ready=True)


def teardown_server(server: tritonserver.Server) -> None:
    server.stop()


def setup_service(
    server: tritonserver.Server,
    frontend: Union[KServeHttp, KServeGrpc, Metrics],
    options=None,
) -> Union[KServeHttp, KServeGrpc, Metrics]:

    service = frontend(server=server, options=options)
    service.start()
    return service


def teardown_service(service: Union[KServeHttp, KServeGrpc]) -> None:
    service.stop()


def setup_client(
    frontend_client: Union["tritonclient.http", "tritonclient.grpc"], url: str
):

    return frontend_client.InferenceServerClient(url=url)


def teardown_client(
    client: Union[
        "tritonclient.http.InferenceServerClient",
        "tritonclient.grpc.InferenceServerClient",
    ],
) -> None:
    client.close()


def send_and_test_inference_identity(
    frontend_client: Union[
        "tritonclient.http.InferenceServerClient",
        "tritonclient.grpc.InferenceServerClient",
    ],
    url: str,
) -> bool:

    model_name = "identity"
    client = setup_client(frontend_client, url)
    input_data = np.array(["testing"], dtype=object)

    inputs = [frontend_client.InferInput("INPUT0", input_data.shape, "BYTES")]
    outputs = [frontend_client.InferRequestedOutput("OUTPUT0")]

    inputs[0].set_data_from_numpy(input_data)

    results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    output_data = results.as_numpy("OUTPUT0")

    teardown_client(client)
    return input_data[0] == output_data[0].decode()


def send_and_test_stream_inference(
    frontend_client: Union[
        "tritonclient.http.InferenceServerClient",
        "tritonclient.grpc.InferenceServerClient",
    ],
    url: str,
) -> bool:

    num_requests = 100
    requests = []
    for i in range(num_requests):
        input0_np = np.array([[float(i) / 1000]], dtype=np.float32)
        inputs = [frontend_client.InferInput("INPUT0", input0_np.shape, "FP32")]
        inputs[0].set_data_from_numpy(input0_np)
        requests.append(inputs)

    responses = []

    def callback(responses, result, error):
        responses.append({"result": result, "error": error})

    client = frontend_client.InferenceServerClient(url=url)
    client.start_stream(partial(callback, responses))
    for inputs in requests:
        client.async_stream_infer("delayed_identity", inputs)
    client.stop_stream()
    teardown_client(client)

    assert len(responses) == num_requests
    for i in range(len(responses)):
        assert responses[i]["error"] is None
        output0_np = responses[i]["result"].as_numpy(name="OUTPUT0")
        assert np.allclose(output0_np, [[float(i) / 1000]])

    return True


def send_and_test_generate_inference() -> bool:

    model_name = "identity"
    url = f"http://localhost:8000/v2/models/{model_name}/generate"
    input_text = "testing"
    data = {
        "INPUT0": input_text,
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        output_text = result.get("OUTPUT0", "")

        if output_text == input_text:
            return True

    return False


def get_metrics(metrics_url: str, model_name: str = "identity") -> Tuple[int, int]:

    response = requests.get(metrics_url)
    inference_count = None

    if response.status_code == 200:
        inference_count = _extract_inference_count(response.text, model_name)
    return response.status_code, inference_count


def _extract_inference_count(metrics_data: str, model_name: str):

    pattern = (
        rf'nv_inference_count\{{.*?model="{re.escape(model_name)}".*?\}}\s+([0-9.]+)'
    )
    match = re.search(pattern, metrics_data)
    if match:
        return int(float(match.group(1)))

    return None
