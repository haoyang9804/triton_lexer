import unittest

import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *


class TestGrpcAioClient(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self._triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

    async def asyncTearDown(self):
        await self._triton_client.close()

    async def test_is_server_live(self):
        ret = await self._triton_client.is_server_live()
        self.assertEqual(ret, True)

    async def test_is_server_ready(self):
        ret = await self._triton_client.is_server_ready()
        self.assertEqual(ret, True)

    async def test_is_model_ready(self):
        ret = await self._triton_client.is_model_ready("simple")
        self.assertEqual(ret, True)

    async def test_get_server_metadata(self):
        ret = await self._triton_client.get_server_metadata()
        self.assertEqual(ret.name, "triton")

        ret = await self._triton_client.get_server_metadata(as_json=True)
        self.assertEqual(ret["name"], "triton")

    async def test_get_model_metadata(self):
        ret = await self._triton_client.get_model_metadata("simple")
        self.assertEqual(ret.name, "simple")

    async def test_get_model_config(self):
        ret = await self._triton_client.get_model_config("simple")
        self.assertEqual(ret.config.name, "simple")

    async def test_get_model_repository_index(self):
        ret = await self._triton_client.get_model_repository_index()
        self.assertEqual(len(ret.models), 8)

    async def test_load_model(self):
        with self.assertRaisesRegex(
            InferenceServerException,
            "\[StatusCode\.UNAVAILABLE\] explicit model load / unload is not allowed if polling is enabled",
        ):
            await self._triton_client.load_model("simple")

    async def test_unload_model(self):
        with self.assertRaisesRegex(
            InferenceServerException,
            "\[StatusCode\.UNAVAILABLE\] explicit model load / unload is not allowed if polling is enabled",
        ):
            await self._triton_client.load_model("simple")

    async def test_get_inference_statistics(self):
        await self._triton_client.get_inference_statistics()

    async def test_update_trace_settings(self):
        await self._triton_client.update_trace_settings()

    async def test_get_trace_settings(self):
        await self._triton_client.get_trace_settings()

    async def test_get_system_shared_memory_status(self):
        await self._triton_client.get_system_shared_memory_status()

    async def test_register_system_shared_memory(self):
        with self.assertRaisesRegex(
            InferenceServerException,
            "\[StatusCode\.INTERNAL\] Unable to open shared memory region: ''",
        ):
            await self._triton_client.register_system_shared_memory("", "", 0)

    async def test_unregister_system_shared_memory(self):
        await self._triton_client.unregister_system_shared_memory()

    async def test_get_cuda_shared_memory_status(self):
        await self._triton_client.get_cuda_shared_memory_status()

    async def test_register_cuda_shared_memory(self):
        with self.assertRaisesRegex(
            InferenceServerException,
            "failed to register shared memory region.*invalid args",
        ):
            await self._triton_client.register_cuda_shared_memory("", b"", 0, 0)

    async def test_unregister_cuda_shared_memory(self):
        await self._triton_client.unregister_cuda_shared_memory()


if __name__ == "__main__":
    unittest.main()
