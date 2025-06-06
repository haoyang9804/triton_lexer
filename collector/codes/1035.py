import json
import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as tritongrpcclient
import tritonclient.grpc.aio as asynctritongrpcclient
from tritonclient.grpc import InferenceServerClientPlugin
from tritonclient.utils import np_to_triton_dtype


class TestPlugin(InferenceServerClientPlugin):
    def __init__(self, headers):
        self._headers = headers

    def __call__(self, request):
        request.headers.update(self._headers)


def prepare_infer_inputs(headers):
    expected_headers = np.array([json.dumps(headers)], dtype=object)
    inputs = []
    inputs.append(
        tritongrpcclient.InferInput(
            "EXPECTED_HEADERS",
            expected_headers.shape,
            np_to_triton_dtype(expected_headers.dtype),
        )
    )
    inputs[0].set_data_from_numpy(expected_headers)

    return inputs


class GRPCClientPluginAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._headers = {"my-key": "my-value"}
        self._plugin = TestPlugin(self._headers)
        self._client = asynctritongrpcclient.InferenceServerClient(url="localhost:8001")

    async def test_simple_infer(self):
        model = "client_plugin_test"
        inputs = prepare_infer_inputs(self._headers)
        self._client.register_plugin(self._plugin)
        response = await self._client.infer(model_name=model, inputs=inputs)
        test_success = response.as_numpy("TEST_SUCCESS")
        self.assertEqual(test_success, True)

        self._client.unregister_plugin()
        inputs = prepare_infer_inputs({})
        response = await self._client.infer(model_name=model, inputs=inputs)
        test_success = response.as_numpy("TEST_SUCCESS")
        self.assertEqual(test_success, True)

    async def asyncTearDown(self):
        await self._client.close()


class GRPCClientPluginTest(tu.TestResultCollector):
    def setUp(self):
        self._headers = {"my-key": "my-value"}
        self._plugin = TestPlugin(self._headers)
        self._client = tritongrpcclient.InferenceServerClient(url="localhost:8001")

    def test_simple_infer(self):

        model = "client_plugin_test"
        inputs = prepare_infer_inputs(self._headers)
        self._client.register_plugin(self._plugin)
        self.assertEqual(self._plugin, self._client.plugin())
        response = self._client.infer(model_name=model, inputs=inputs)
        test_success = response.as_numpy("TEST_SUCCESS")
        self.assertEqual(test_success, True)

        inputs = prepare_infer_inputs({})
        self._client.unregister_plugin()
        self.assertEqual(None, self._client.plugin())
        response = self._client.infer(model_name=model, inputs=inputs)
        test_success = response.as_numpy("TEST_SUCCESS")
        self.assertEqual(test_success, True)

    def tearDown(self):
        self._client.close()


if __name__ == "__main__":
    unittest.main()
