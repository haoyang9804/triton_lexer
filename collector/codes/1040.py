import sys

sys.path.append("../common")

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import test_util as tu
import tritonclient.http as tritonhttpclient
import tritonclient.http.aio as asynctritonhttpclient
from tritonclient.http import InferenceServerClientPlugin
from tritonclient.utils import np_to_triton_dtype


class TestPlugin(InferenceServerClientPlugin):
    def __init__(self, headers):
        self._headers = headers

    def __call__(self, request):
        request.headers.update(self._headers)


class HTTPClientPluginAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._headers = {"MY-KEY": "MY-VALUE"}
        self._plugin = TestPlugin(self._headers)
        self._client = asynctritonhttpclient.InferenceServerClient(url="localhost:8001")

    async def test_server_is_live(self):

        self._client._stub.get = AsyncMock()

        self._client.register_plugin(self._plugin)
        self.assertEqual(self._plugin, self._client.plugin())
        await self._client.is_server_live()
        self._client._stub.get.assert_awaited_with(
            url=unittest.mock.ANY, headers=self._headers
        )

        self._client.unregister_plugin()
        self.assertEqual(None, self._client.plugin())
        await self._client.is_server_live()
        self._client._stub.get.assert_awaited_with(url=unittest.mock.ANY, headers={})

    async def test_simple_infer(self):

        post_return = MagicMock()
        post_return.read = AsyncMock()
        self._client._stub.post = AsyncMock(return_value=post_return)

        np_input = np.arange(8, dtype=np.float32).reshape(1, -1)
        model = "onnx_zero_1_float32"

        inputs = []
        inputs.append(
            tritonhttpclient.InferInput(
                "INPUT0", np_input.shape, np_to_triton_dtype(np_input.dtype)
            )
        )

        inputs[0].set_data_from_numpy(np_input, binary_data=False)

        async def run_infer(headers):
            with patch("tritonclient.http.aio._raise_if_error"):
                with patch("tritonclient.http.aio.InferResult"):
                    await self._client.infer(model_name=model, inputs=inputs)
                    self._client._stub.post.assert_awaited_with(
                        url=unittest.mock.ANY, data=unittest.mock.ANY, headers=headers
                    )

        self._client.register_plugin(self._plugin)
        await run_infer(self._headers)

        self._client.unregister_plugin()
        await run_infer({})

    async def asyncTearDown(self):
        await self._client.close()


class HTTPClientPluginTest(tu.TestResultCollector):
    def setUp(self):
        self._headers = {"MY-KEY": "MY-VALUE"}
        self._plugin = TestPlugin(self._headers)
        self._client = tritonhttpclient.InferenceServerClient(url="localhost:8001")

        self._client._client_stub = MagicMock()

    def test_server_is_live(self):

        self._client.register_plugin(self._plugin)
        self._client.is_server_live()
        self._client._client_stub.get.assert_called_with(
            unittest.mock.ANY, headers=self._headers
        )

        self._client.unregister_plugin()
        self._client.is_server_live()
        self._client._client_stub.get.assert_called_with(unittest.mock.ANY, headers={})

    def test_simple_infer(self):
        np_input = np.arange(8, dtype=np.float32).reshape(1, -1)
        model = "onnx_zero_1_float32"

        inputs = []
        inputs.append(
            tritonhttpclient.InferInput(
                "INPUT0", np_input.shape, np_to_triton_dtype(np_input.dtype)
            )
        )

        inputs[0].set_data_from_numpy(np_input, binary_data=False)

        def run_infer(headers):
            with patch("tritonclient.http._client._raise_if_error"):
                with patch("tritonclient.http._client.InferResult"):
                    self._client.infer(model_name=model, inputs=inputs)
                    self._client._client_stub.post.assert_called_with(
                        request_uri=unittest.mock.ANY,
                        body=unittest.mock.ANY,
                        headers=headers,
                    )

        self._client.register_plugin(self._plugin)
        run_infer(self._headers)

        self._client.unregister_plugin()
        run_infer({})

    def tearDown(self):
        self._client.close()


if __name__ == "__main__":
    unittest.main()
