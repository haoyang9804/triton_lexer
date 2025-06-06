import sys
import unittest

sys.path.append("../common")

import test_util as tu
import tritonclient.grpc as tritongrpcclient
import tritonclient.grpc.aio as asynctritongrpcclient
from tritonclient.grpc.aio.auth import BasicAuth as AsyncBasicAuth
from tritonclient.grpc.auth import BasicAuth


class GRPCBasicAuthTest(tu.TestResultCollector):
    def setUp(self):

        self._client = tritongrpcclient.InferenceServerClient(url="localhost:8004")
        self._client.register_plugin(BasicAuth("username", "password"))

    def test_client_call(self):
        self.assertTrue(self._client.is_server_live())

    def tearDown(self):
        self._client.close()


class GRPCBasicAuthAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        self._client = asynctritongrpcclient.InferenceServerClient(url="localhost:8004")
        self._client.register_plugin(AsyncBasicAuth("username", "password"))

    async def test_client_call(self):
        self.assertTrue(await self._client.is_server_live())

    async def asyncTearDown(self):
        await self._client.close()


if __name__ == "__main__":
    unittest.main()
