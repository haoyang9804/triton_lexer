import logging

import gevent.ssl
import tritonclient.http as httpclient

from .client import TritonClient


class TritonHTTPClient(TritonClient):

    def __init__(self, server_url, ssl_options={}):

        ssl = False
        client_ssl_options = {}
        ssl_context_factory = gevent.ssl._create_unverified_context
        insecure = True
        verify_peer = 0
        verify_host = 0

        if server_url.startswith("http://"):
            server_url = server_url.replace("http://", "", 1)
        elif server_url.startswith("https://"):
            ssl = True
            server_url = server_url.replace("https://", "", 1)
        if "ssl-https-ca-certificates-file" in ssl_options:
            client_ssl_options["ca_certs"] = ssl_options[
                "ssl-https-ca-certificates-file"
            ]
        if "ssl-https-client-certificate-file" in ssl_options:
            if (
                "ssl-https-client-certificate-type" in ssl_options
                and ssl_options["ssl-https-client-certificate-type"] == "PEM"
            ):
                client_ssl_options["certfile"] = ssl_options[
                    "ssl-https-client-certificate-file"
                ]
            else:
                logging.warning(
                    "model-analyzer with SSL must be passed a client certificate file in PEM format."
                )
        if "ssl-https-private-key-file" in ssl_options:
            if (
                "ssl-https-private-key-type" in ssl_options
                and ssl_options["ssl-https-private-key-type"] == "PEM"
            ):
                client_ssl_options["keyfile"] = ssl_options[
                    "ssl-https-private-key-file"
                ]
            else:
                logging.warning(
                    "model-analyzer with SSL must be passed a private key file in PEM format."
                )
        if "ssl-https-verify-peer" in ssl_options:
            verify_peer = ssl_options["ssl-https-verify-peer"]
        if "ssl-https-verify-host" in ssl_options:
            verify_host = ssl_options["ssl-https-verify-host"]
        if verify_peer != 0 and verify_host != 0:
            ssl_context_factory = None
            insecure = False

        self._client = httpclient.InferenceServerClient(
            url=server_url,
            ssl=ssl,
            ssl_options=client_ssl_options,
            ssl_context_factory=ssl_context_factory,
            insecure=insecure,
        )

    def get_model_repository_index(self):

        return self._client.get_model_repository_index()

    def is_model_ready(self, model_name: str) -> bool:

        return self._client.is_model_ready(model_name)
