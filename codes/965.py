import tritonclient.grpc as grpcclient

from .client import TritonClient


class TritonGRPCClient(TritonClient):

    def __init__(self, server_url, ssl_options={}):

        ssl = False
        root_certificates = None
        private_key = None
        certificate_chain = None

        if "ssl-grpc-use-ssl" in ssl_options:
            ssl = ssl_options["ssl-grpc-use-ssl"].lower() == "true"
        if "ssl-grpc-root-certifications-file" in ssl_options:
            root_certificates = ssl_options["ssl-grpc-root-certifications-file"]
        if "ssl-grpc-private-key-file" in ssl_options:
            private_key = ssl_options["ssl-grpc-private-key-file"]
        if "ssl-grpc-certificate-chain-file" in ssl_options:
            certificate_chain = ssl_options["ssl-grpc-certificate-chain-file"]

        self._client = grpcclient.InferenceServerClient(
            url=server_url,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )

    def get_model_config(self, model_name, num_retries):

        self.wait_for_model_ready(model_name, num_retries)
        model_config_dict = self._client.get_model_config(model_name, as_json=True)
        return model_config_dict["config"]

    def get_model_repository_index(self):

        return self._client.get_model_repository_index(as_json=True)["models"]

    def is_model_ready(self, model_name: str) -> bool:

        return self._client.is_model_ready(model_name)
