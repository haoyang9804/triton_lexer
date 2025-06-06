import sys

sys.path.append("../common")

import base64
import os
import time
import unittest
from functools import partial

import infer_util as iu
import numpy as np
import requests
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.utils.cuda_shared_memory as cshm
from tritonclient.utils import *


class CudaSharedMemoryTestBase(tu.TestResultCollector):
    DEFAULT_SHM_BYTE_SIZE = 64

    def setUp(self):
        self._setup_client()
        self._shm_handles = []

    def tearDown(self):
        self._cleanup_shm_handles()

    def _setup_client(self):
        self.protocol = os.environ.get("CLIENT_TYPE", "http")
        if self.protocol == "http":
            self.url = "localhost:8000"
            self.triton_client = httpclient.InferenceServerClient(
                self.url, verbose=True
            )
        else:
            self.url = "localhost:8001"
            self.triton_client = grpcclient.InferenceServerClient(
                self.url, verbose=True
            )

    def _configure_server(
        self,
        create_byte_size=DEFAULT_SHM_BYTE_SIZE,
        register_byte_size=DEFAULT_SHM_BYTE_SIZE,
        device_id=0,
    ):

        self._cleanup_shm_handles()
        shm_ip0_handle = cshm.create_shared_memory_region(
            "input0_data", create_byte_size, device_id
        )
        shm_ip1_handle = cshm.create_shared_memory_region(
            "input1_data", create_byte_size, device_id
        )
        shm_op0_handle = cshm.create_shared_memory_region(
            "output0_data", create_byte_size, device_id
        )
        shm_op1_handle = cshm.create_shared_memory_region(
            "output1_data", create_byte_size, device_id
        )

        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        cshm.set_shared_memory_region(shm_ip0_handle, [input0_data])
        cshm.set_shared_memory_region(shm_ip1_handle, [input1_data])

        self.triton_client.register_cuda_shared_memory(
            "input0_data",
            cshm.get_raw_handle(shm_ip0_handle),
            device_id,
            register_byte_size,
        )
        self.triton_client.register_cuda_shared_memory(
            "input1_data",
            cshm.get_raw_handle(shm_ip1_handle),
            device_id,
            register_byte_size,
        )
        self.triton_client.register_cuda_shared_memory(
            "output0_data",
            cshm.get_raw_handle(shm_op0_handle),
            device_id,
            register_byte_size,
        )
        self.triton_client.register_cuda_shared_memory(
            "output1_data",
            cshm.get_raw_handle(shm_op1_handle),
            device_id,
            register_byte_size,
        )
        self._shm_handles = [
            shm_ip0_handle,
            shm_ip1_handle,
            shm_op0_handle,
            shm_op1_handle,
        ]
        self.shm_names = ["input0_data", "input1_data", "output0_data", "output1_data"]

    def _cleanup_shm_handles(self):
        for shm_handle in self._shm_handles:
            cshm.destroy_shared_memory_region(shm_handle)
        self._shm_handles = []


class CudaSharedMemoryTest(CudaSharedMemoryTestBase):
    def test_invalid_create_shm(self):

        with self.assertRaisesRegex(
            cshm.CudaSharedMemoryException, "unable to create cuda shared memory handle"
        ):
            self._shm_handles.append(
                cshm.create_shared_memory_region("dummy_data", -1, 0)
            )

    def test_valid_create_set_register(self):

        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        cshm.set_shared_memory_region(
            shm_op0_handle, [np.array([1, 2], dtype=np.float32)]
        )
        self.triton_client.register_cuda_shared_memory(
            "dummy_data", cshm.get_raw_handle(shm_op0_handle), 0, 8
        )
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 1)
        else:
            self.assertEqual(len(shm_status.regions), 1)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_before_register(self):

        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        self.triton_client.unregister_cuda_shared_memory("dummy_data")
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 0)
        else:
            self.assertEqual(len(shm_status.regions), 0)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_after_register(self):

        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        self.triton_client.register_cuda_shared_memory(
            "dummy_data", cshm.get_raw_handle(shm_op0_handle), 0, 8
        )
        self.triton_client.unregister_cuda_shared_memory("dummy_data")
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 0)
        else:
            self.assertEqual(len(shm_status.regions), 0)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_reregister_after_register(self):

        shm_op0_handle = cshm.create_shared_memory_region("dummy_data", 8, 0)
        self.triton_client.register_cuda_shared_memory(
            "dummy_data", cshm.get_raw_handle(shm_op0_handle), 0, 8
        )
        try:
            self.triton_client.register_cuda_shared_memory(
                "dummy_data", cshm.get_raw_handle(shm_op0_handle), 0, 8
            )
        except Exception as ex:
            self.assertIn(
                "shared memory region 'dummy_data' already in manager", str(ex)
            )
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 1)
        else:
            self.assertEqual(len(shm_status.regions), 1)
        cshm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_after_inference(self):

        error_msg = []
        self._configure_server()
        iu.shm_basic_infer(
            self,
            self.triton_client,
            self._shm_handles[0],
            self._shm_handles[1],
            self._shm_handles[2],
            self._shm_handles[3],
            error_msg,
            protocol=self.protocol,
            use_cuda_shared_memory=True,
        )
        if len(error_msg) > 0:
            raise Exception(str(error_msg))

        self.triton_client.unregister_cuda_shared_memory("output0_data")
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 3)
        else:
            self.assertEqual(len(shm_status.regions), 3)
        self._cleanup_shm_handles()

    def test_register_after_inference(self):

        error_msg = []
        self._configure_server()
        iu.shm_basic_infer(
            self,
            self.triton_client,
            self._shm_handles[0],
            self._shm_handles[1],
            self._shm_handles[2],
            self._shm_handles[3],
            error_msg,
            protocol=self.protocol,
            use_cuda_shared_memory=True,
        )
        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 64, 0)
        self.triton_client.register_cuda_shared_memory(
            "input2_data", cshm.get_raw_handle(shm_ip2_handle), 0, 64
        )
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(shm_status), 5)
        else:
            self.assertEqual(len(shm_status.regions), 5)
        self._shm_handles.append(shm_ip2_handle)
        self._cleanup_shm_handles()

    def test_too_big_shm(self):

        error_msg = []
        self._configure_server()
        shm_ip2_handle = cshm.create_shared_memory_region("input2_data", 128, 0)
        self.triton_client.register_cuda_shared_memory(
            "input2_data", cshm.get_raw_handle(shm_ip2_handle), 0, 128
        )
        iu.shm_basic_infer(
            self,
            self.triton_client,
            self._shm_handles[0],
            shm_ip2_handle,
            self._shm_handles[2],
            self._shm_handles[3],
            error_msg,
            big_shm_name="input2_data",
            big_shm_size=128,
            protocol=self.protocol,
            use_cuda_shared_memory=True,
        )
        if len(error_msg) > 0:
            self.assertIn(
                "input byte size mismatch for input 'INPUT1' for model 'simple'. Expected 64, got 128",
                error_msg[-1],
            )
        self._shm_handles.append(shm_ip2_handle)
        self._cleanup_shm_handles()

    def test_mixed_raw_shm(self):

        error_msg = []
        self._configure_server()
        input1_data = np.ones(shape=16, dtype=np.int32)
        iu.shm_basic_infer(
            self,
            self.triton_client,
            self._shm_handles[0],
            [input1_data],
            self._shm_handles[2],
            self._shm_handles[3],
            error_msg,
            protocol=self.protocol,
            use_cuda_shared_memory=True,
        )

        if len(error_msg) > 0:
            raise Exception(error_msg[-1])
        self._cleanup_shm_handles()

    def test_unregisterall(self):

        self._configure_server()
        status_before = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(status_before), 4)
        else:
            self.assertEqual(len(status_before.regions), 4)
        self.triton_client.unregister_cuda_shared_memory()
        status_after = self.triton_client.get_cuda_shared_memory_status()
        if self.protocol == "http":
            self.assertEqual(len(status_after), 0)
        else:
            self.assertEqual(len(status_after.regions), 0)
        self._cleanup_shm_handles()

    def test_register_out_of_bound(self):
        create_byte_size = self.DEFAULT_SHM_BYTE_SIZE

        with self.assertRaisesRegex(
            InferenceServerException,
            "failed to register shared memory region.*invalid args",
        ):
            self._configure_server(
                create_byte_size=create_byte_size,
                register_byte_size=create_byte_size + 1,
            )

    def test_infer_offset_out_of_bound(self):

        error_msg = []
        self._configure_server()
        if self.protocol == "http":

            offset = 2**64 - 32
        else:

            offset = 64
        iu.shm_basic_infer(
            self,
            self.triton_client,
            self._shm_handles[0],
            self._shm_handles[1],
            self._shm_handles[2],
            self._shm_handles[3],
            error_msg,
            shm_output_offset=offset,
            protocol=self.protocol,
            use_system_shared_memory=False,
            use_cuda_shared_memory=True,
        )

        self.assertEqual(len(error_msg), 1)
        self.assertIn("Invalid offset for shared memory region", error_msg[0])
        self._cleanup_shm_handles()

    def test_infer_byte_size_out_of_bound(self):

        error_msg = []
        self._configure_server()
        offset = 60
        byte_size = self.DEFAULT_SHM_BYTE_SIZE

        iu.shm_basic_infer(
            self,
            self.triton_client,
            self._shm_handles[0],
            self._shm_handles[1],
            self._shm_handles[2],
            self._shm_handles[3],
            error_msg,
            shm_output_offset=offset,
            shm_output_byte_size=byte_size,
            protocol=self.protocol,
            use_system_shared_memory=False,
            use_cuda_shared_memory=True,
        )
        self.assertEqual(len(error_msg), 1)
        self.assertIn(
            "Invalid offset + byte size for shared memory region", error_msg[0]
        )
        self._cleanup_shm_handles()


def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)


class TestCudaSharedMemoryUnregister(CudaSharedMemoryTestBase):
    def _create_request_data(self):
        self.triton_client.unregister_cuda_shared_memory()
        self._configure_server()

        if self.protocol == "http":
            inputs = [
                httpclient.InferInput("INPUT0", [1, 16], "INT32"),
                httpclient.InferInput("INPUT1", [1, 16], "INT32"),
            ]
            outputs = [
                httpclient.InferRequestedOutput("OUTPUT0", binary_data=True),
                httpclient.InferRequestedOutput("OUTPUT1", binary_data=False),
            ]
        else:
            inputs = [
                grpcclient.InferInput("INPUT0", [1, 16], "INT32"),
                grpcclient.InferInput("INPUT1", [1, 16], "INT32"),
            ]
            outputs = [
                grpcclient.InferRequestedOutput("OUTPUT0"),
                grpcclient.InferRequestedOutput("OUTPUT1"),
            ]

        inputs[0].set_shared_memory("input0_data", self.DEFAULT_SHM_BYTE_SIZE)
        inputs[1].set_shared_memory("input1_data", self.DEFAULT_SHM_BYTE_SIZE)
        outputs[0].set_shared_memory("output0_data", self.DEFAULT_SHM_BYTE_SIZE)
        outputs[1].set_shared_memory("output1_data", self.DEFAULT_SHM_BYTE_SIZE)

        return inputs, outputs

    def _test_unregister_shm_request_pass(self):
        self._test_shm_found()

        with httpclient.InferenceServerClient(
            "localhost:8000", verbose=True
        ) as second_client:
            second_client.unregister_cuda_shared_memory()

        self._test_shm_found()

    def _test_shm_not_found(self):
        second_client = httpclient.InferenceServerClient("localhost:8000", verbose=True)

        for shm_name in self.shm_names:
            with self.assertRaises(InferenceServerException) as ex:
                second_client.get_cuda_shared_memory_status(shm_name)
                self.assertIn(
                    f"Unable to find cuda shared memory region: '{shm_name}'",
                    str(ex.exception),
                )

    def _test_shm_found(self):
        second_client = httpclient.InferenceServerClient("localhost:8000", verbose=True)

        status = second_client.get_cuda_shared_memory_status()
        self.assertEqual(len(status), len(self.shm_names))

        for shm_info in status:
            self.assertIn(shm_info["name"], self.shm_names)

    def test_unregister_shm_during_inference_single_req_http(self):
        inputs, outputs = self._create_request_data()

        async_request = self.triton_client.async_infer(
            model_name="simple", inputs=inputs, outputs=outputs
        )

        time.sleep(2)

        self._test_unregister_shm_request_pass()

        async_request.get_result()

        self._test_shm_not_found()

    def test_unregister_shm_during_inference_multiple_req_http(self):
        inputs, outputs = self._create_request_data()

        async_request = self.triton_client.async_infer(
            model_name="simple", inputs=inputs, outputs=outputs
        )

        time.sleep(2)

        self._test_unregister_shm_request_pass()
        time.sleep(2)

        second_client = httpclient.InferenceServerClient("localhost:8000", verbose=True)
        second_async_request = second_client.async_infer(
            model_name="simple", inputs=inputs, outputs=outputs
        )

        async_request.get_result()

        self._test_shm_found()

        second_async_request.get_result()

        self._test_shm_not_found()

    def test_unregister_shm_after_inference_http(self):
        inputs, outputs = self._create_request_data()

        async_request = self.triton_client.async_infer(
            model_name="simple", inputs=inputs, outputs=outputs
        )

        time.sleep(2)

        self._test_shm_found()

        async_request.get_result()

        self._test_shm_found()

        self.triton_client.unregister_cuda_shared_memory()
        self._test_shm_not_found()

    def test_unregister_shm_during_inference_single_req_grpc(self):
        inputs, outputs = self._create_request_data()
        user_data = []

        self.triton_client.async_infer(
            model_name="simple",
            inputs=inputs,
            outputs=outputs,
            callback=partial(callback, user_data),
        )

        time.sleep(2)

        self._test_unregister_shm_request_pass()

        time_out = 20
        while (len(user_data) == 0) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)
        time.sleep(2)

        self._test_shm_not_found()

    def test_unregister_shm_during_inference_multiple_req_grpc(self):
        inputs, outputs = self._create_request_data()
        user_data = []

        self.triton_client.async_infer(
            model_name="simple",
            inputs=inputs,
            outputs=outputs,
            callback=partial(callback, user_data),
        )

        time.sleep(2)

        self._test_unregister_shm_request_pass()

        second_user_data = []
        second_client = grpcclient.InferenceServerClient("localhost:8001", verbose=True)
        second_client.async_infer(
            model_name="simple",
            inputs=inputs,
            outputs=outputs,
            callback=partial(callback, second_user_data),
        )

        time_out = 10
        while (len(user_data) == 0) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)
        time.sleep(2)

        self._test_shm_found()

        time_out = 20
        while (len(second_user_data) == 0) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)
        time.sleep(2)

        self._test_shm_not_found()

    def test_unregister_shm_after_inference_grpc(self):
        inputs, outputs = self._create_request_data()
        user_data = []

        self.triton_client.async_infer(
            model_name="simple",
            inputs=inputs,
            outputs=outputs,
            callback=partial(callback, user_data),
        )

        time.sleep(2)

        self._test_shm_found()

        time_out = 20
        while (len(user_data) == 0) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)
        time.sleep(2)

        self._test_shm_found()

        self.triton_client.unregister_cuda_shared_memory()
        self._test_shm_not_found()


class CudaSharedMemoryTestRawHttpRequest(unittest.TestCase):
    def setUp(self):
        self.url = "localhost:8000"
        self.client = httpclient.InferenceServerClient(url=self.url, verbose=True)
        self.valid_shm_handle = None

    def tearDown(self):
        self.client.unregister_cuda_shared_memory()
        if self.valid_shm_handle:
            cshm.destroy_shared_memory_region(self.valid_shm_handle)
        self.client.close()

    def _generate_mock_base64_raw_handle(self, data_length):
        original_data_length = data_length * 3 // 4
        random_data = b"A" * original_data_length
        encoded_data = base64.b64encode(random_data)

        assert (
            len(encoded_data) == data_length
        ), "Encoded data length does not match the required length."
        return encoded_data

    def _send_register_cshm_request(self, raw_handle, device_id, byte_size, shm_name):
        cuda_shared_memory_register_request = {
            "raw_handle": {"b64": raw_handle.decode("utf-8")},
            "device_id": device_id,
            "byte_size": byte_size,
        }

        url = "http://{}/v2/cudasharedmemory/region/{}/register".format(
            self.url, shm_name
        )
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            url, headers=headers, json=cuda_shared_memory_register_request
        )
        return response

    def test_exceeds_cshm_handle_size_limit(self):

        byte_size = 1 << 31
        device_id = 0
        shm_name = "invalid_shm"

        raw_handle = self._generate_mock_base64_raw_handle(byte_size)
        response = self._send_register_cshm_request(
            raw_handle, device_id, byte_size, shm_name
        )
        self.assertNotEqual(response.status_code, 200)

        try:
            error_message = response.json().get("error", "")
            self.assertIn(
                "'raw_handle' exceeds the maximum allowed data size limit INT_MAX",
                error_message,
            )
        except ValueError:
            self.fail("Response is not valid JSON")

    def test_invalid_small_cshm_handle(self):
        byte_size = 64
        device_id = 0
        shm_name = "invalid_shm"

        raw_handle = self._generate_mock_base64_raw_handle(byte_size)
        response = self._send_register_cshm_request(
            raw_handle, device_id, byte_size, shm_name
        )
        self.assertNotEqual(response.status_code, 200)

        try:
            error_message = response.json().get("error", "")
            self.assertIn(
                "'raw_handle' must be a valid base64 encoded cudaIpcMemHandle_t",
                error_message,
            )
        except ValueError:
            self.fail("Response is not valid JSON")

    def test_valid_cshm_handle(self):
        byte_size = 64
        device_id = 0
        shm_name = "test_shm"

        self.valid_shm_handle = cshm.create_shared_memory_region(
            shm_name, byte_size, device_id
        )
        raw_handle = cshm.get_raw_handle(self.valid_shm_handle)

        response = self._send_register_cshm_request(
            raw_handle, device_id, byte_size, shm_name
        )
        self.assertEqual(response.status_code, 200)

        status = self.client.get_cuda_shared_memory_status()
        self.assertEqual(len(status), 1)
        self.assertEqual(status[0]["name"], shm_name)


if __name__ == "__main__":
    unittest.main()
