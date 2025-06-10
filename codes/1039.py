import sys
import unittest

sys.path.append("../common")

import test_util as tu
import tritonclient.http as tritonhttpclient
import tritonclient.http.aio as asynctritonhttpclient
from tritonclient.http.aio.auth import BasicAuth as AsyncBasicAuth
from tritonclient.http.auth import BasicAuth


class HTTPBasicAuthTest(tu.TestResultCollector):
    def setUp(self):

        self._client = tritonhttpclient.InferenceServerClient(url="localhost:8004")
        self._client.register_plugin(BasicAuth("username", "password"))

    def test_client_call(self):
        self.assertTrue(self._client.is_server_live())

    def tearDown(self):
        self._client.close()


class HTTPBasicAuthAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        self._client = asynctritonhttpclient.InferenceServerClient(url="localhost:8004")
        self._client.register_plugin(AsyncBasicAuth("username", "password"))

    async def test_client_call(self):
        self.assertTrue(await self._client.is_server_live())

    async def asyncTearDown(self):
        await self._client.close()


if __name__ == "__main__":
    unittest.main()
