import pathlib

import numpy as np
import tritonclient.http as httpclient
import tritonserver
from tritonfrontend import KServeHttp


def main():

    model_path = f"{pathlib.Path(__file__).parent.resolve()}/example_model_repository"

    server_options = tritonserver.Options(
        server_id="ExampleServer",
        model_repository=model_path,
        log_error=True,
        log_info=True,
        log_warn=True,
    )

    server = tritonserver.Server(server_options).start(wait_until_ready=True)

    http_options = KServeHttp.Options(port=8005)

    with KServeHttp(server, http_options) as http_service:

        model_name = "identity"
        url = "localhost:8005"

        client = httpclient.InferenceServerClient(url=url)

        input_data = np.array([["Roger Roger"]], dtype=object)

        inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]

        inputs[0].set_data_from_numpy(input_data)

        results = client.infer(model_name, inputs=inputs)

        output_data = results.as_numpy("OUTPUT0")

        print("--------------------- INFERENCE RESULTS ---------------------")
        print("Output data:", output_data)
        print("-------------------------------------------------------------")

    server.stop()


if __name__ == "__main__":
    main()
