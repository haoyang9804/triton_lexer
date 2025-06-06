import sys

sys.path.append("../common")

import os
import time
import unittest
from functools import partial

import infer_util as iu
import numpy as np
import psutil
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.utils.shared_memory as shm
from tritonclient import utils


class SystemSharedMemoryTestBase(tu.TestResultCollector):
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
        register_offset=0,
    ):

        self._cleanup_shm_handles()
        shm_ip0_handle = shm.create_shared_memory_region(
            "input0_data", "/input0_data", create_byte_size
        )
        shm_ip1_handle = shm.create_shared_memory_region(
            "input1_data", "/input1_data", create_byte_size
        )
        shm_op0_handle = shm.create_shared_memory_region(
            "output0_data", "/output0_data", create_byte_size
        )
        shm_op1_handle = shm.create_shared_memory_region(
            "output1_data", "/output1_data", create_byte_size
        )
        self._shm_handles = [
            shm_ip0_handle,
            shm_ip1_handle,
            shm_op0_handle,
            shm_op1_handle,
        ]

        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input1_data = np.ones(shape=16, dtype=np.int32)
        shm.set_shared_memory_region(shm_ip0_handle, [input0_data])
        shm.set_shared_memory_region(shm_ip1_handle, [input1_data])
        self.triton_client.register_system_shared_memory(
            "input0_data", "/input0_data", register_byte_size, offset=register_offset
        )
        self.triton_client.register_system_shared_memory(
            "input1_data", "/input1_data", register_byte_size, offset=register_offset
        )
        self.triton_client.register_system_shared_memory(
            "output0_data", "/output0_data", register_byte_size, offset=register_offset
        )
        self.triton_client.register_system_shared_memory(
            "output1_data", "/output1_data", register_byte_size, offset=register_offset
        )
        self.shm_names = ["input0_data", "input1_data", "output0_data", "output1_data"]

    def _cleanup_shm_handles(self):
        for shm_handle in self._shm_handles:
            shm.destroy_shared_memory_region(shm_handle)
        self._shm_handles = []


class SharedMemoryTest(SystemSharedMemoryTestBase):
    def test_invalid_create_shm(self):
        with self.assertRaisesRegex(
            shm.SharedMemoryException, "unable to create the shared memory region"
        ):
            self._shm_handles.append(
                shm.create_shared_memory_region("dummy_data", "/dummy_data", -1)
            )

    def test_valid_create_set_register(self):

        shm_op0_handle = shm.create_shared_memory_region("dummy_data", "/dummy_data", 8)
        shm.set_shared_memory_region(
            shm_op0_handle, [np.array([1, 2], dtype=np.float32)]
        )
        self.triton_client.register_system_shared_memory("dummy_data", "/dummy_data", 8)
        shm_status = self.triton_client.get_system_shared_memory_status()
        if self.protocol == "http":
            self.assertTrue(len(shm_status) == 1)
        else:
            self.assertTrue(len(shm_status.regions) == 1)
        shm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_before_register(self):

        shm_op0_handle = shm.create_shared_memory_region("dummy_data", "/dummy_data", 8)
        self.triton_client.unregister_system_shared_memory("dummy_data")
        shm_status = self.triton_client.get_system_shared_memory_status()
        if self.protocol == "http":
            self.assertTrue(len(shm_status) == 0)
        else:
            self.assertTrue(len(shm_status.regions) == 0)
        shm.destroy_shared_memory_region(shm_op0_handle)

    def test_unregister_after_register(self):

        shm_op0_handle = shm.create_shared_memory_region("dummy_data", "/dummy_data", 8)
        self.triton_client.register_system_shared_memory("dummy_data", "/dummy_data", 8)
        self.triton_client.unregister_system_shared_memory("dummy_data")
        shm_status = self.triton_client.get_system_shared_memory_status()
        if self.protocol == "http":
            self.assertTrue(len(shm_status) == 0)
        else:
            self.assertTrue(len(shm_status.regions) == 0)
        shm.destroy_shared_memory_region(shm_op0_handle)

    def test_reregister_after_register(self):

        shm_op0_handle = shm.create_shared_memory_region("dummy_data", "/dummy_data", 8)
        self.triton_client.register_system_shared_memory("dummy_data", "/dummy_data", 8)
        try:
            self.triton_client.register_system_shared_memory(
                "dummy_data", "/dummy_data", 8
            )
        except Exception as ex:
            self.assertIn(
                "shared memory region 'dummy_data' already in manager", str(ex)
            )
        shm_status = self.triton_client.get_system_shared_memory_status()
        if self.protocol == "http":
            self.assertTrue(len(shm_status) == 1)
        else:
            self.assertTrue(len(shm_status.regions) == 1)
        shm.destroy_shared_memory_region(shm_op0_handle)

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
            use_system_shared_memory=True,
        )
        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        self.triton_client.unregister_system_shared_memory("output0_data")
        shm_status = self.triton_client.get_system_shared_memory_status()
        if self.protocol == "http":
            self.assertTrue(len(shm_status) == 3)
        else:
            self.assertTrue(len(shm_status.regions) == 3)
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
            use_system_shared_memory=True,
        )

        if len(error_msg) > 0:
            raise Exception(str(error_msg))
        shm_ip2_handle = shm.create_shared_memory_region(
            "input2_data", "/input2_data", self.DEFAULT_SHM_BYTE_SIZE
        )
        self.triton_client.register_system_shared_memory(
            "input2_data", "/input2_data", self.DEFAULT_SHM_BYTE_SIZE
        )
        shm_status = self.triton_client.get_system_shared_memory_status()
        if self.protocol == "http":
            self.assertTrue(len(shm_status) == 5)
        else:
            self.assertTrue(len(shm_status.regions) == 5)
        self._shm_handles.append(shm_ip2_handle)
        self._cleanup_shm_handles()

    def test_too_big_shm(self):

        error_msg = []
        self._configure_server()
        shm_ip2_handle = shm.create_shared_memory_region(
            "input2_data", "/input2_data", 128
        )
        self.triton_client.register_system_shared_memory(
            "input2_data", "/input2_data", 128
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
            use_system_shared_memory=True,
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
            use_system_shared_memory=True,
        )
        if len(error_msg) > 0:
            raise Exception(error_msg[-1])
        self._cleanup_shm_handles()

    def test_unregisterall(self):

        self._configure_server()
        status_before = self.triton_client.get_system_shared_memory_status()
        if self.protocol == "http":
            self.assertTrue(len(status_before) == 4)
        else:
            self.assertTrue(len(status_before.regions) == 4)
        self.triton_client.unregister_system_shared_memory()
        status_after = self.triton_client.get_system_shared_memory_status()
        if self.protocol == "http":
            self.assertTrue(len(status_after) == 0)
        else:
            self.assertTrue(len(status_after.regions) == 0)
        self._cleanup_shm_handles()

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
            use_system_shared_memory=True,
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
            use_system_shared_memory=True,
        )
        self.assertEqual(len(error_msg), 1)
        self.assertIn(
            "Invalid offset + byte size for shared memory region", error_msg[0]
        )
        self._cleanup_shm_handles()

    def test_infer_integer_overflow(self):

        error_msg = []
        self._configure_server()

        offset = 32
        byte_size = 2**64 - 32

        if self.protocol == "http":
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
                use_system_shared_memory=True,
            )

            self.assertEqual(len(error_msg), 1)
            self.assertTrue(
                "Integer overflow detected: byte_size " in error_msg[0],
                f"Unexpected error message: {error_msg[0]}",
            )
            self._cleanup_shm_handles()
        else:

            try:
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
                    use_system_shared_memory=True,
                )
                self.assertTrue(
                    False,
                    "Expected gRPC client to fail on value larger than int64_param maximum",
                )
            except ValueError as ex:
                self.assertIn("Value out of range:", str(ex))
            self._cleanup_shm_handles()

    def test_register_out_of_bound(self):
        create_byte_size = self.DEFAULT_SHM_BYTE_SIZE

        with self.assertRaisesRegex(
            utils.InferenceServerException,
            "failed to register shared memory region.*invalid args",
        ):
            self._configure_server(
                create_byte_size=create_byte_size,
                register_byte_size=create_byte_size + 1,
                register_offset=0,
            )

        with self.assertRaisesRegex(
            utils.InferenceServerException,
            "failed to register shared memory region.*invalid args",
        ):
            self._configure_server(
                create_byte_size=create_byte_size,
                register_byte_size=create_byte_size,
                register_offset=1,
            )

        with self.assertRaisesRegex(
            utils.InferenceServerException,
            "failed to register shared memory region.*invalid args",
        ):
            self._configure_server(
                create_byte_size=create_byte_size,
                register_byte_size=1,
                register_offset=create_byte_size,
            )

        with self.assertRaisesRegex(
            utils.InferenceServerException,
            "failed to register shared memory region.*invalid args",
        ):
            self._configure_server(
                create_byte_size=create_byte_size,
                register_byte_size=0,
                register_offset=create_byte_size + 1,
            )

    def test_python_client_leak(self):
        process = psutil.Process()
        initial_mem_usage = process.memory_info().rss / 1024**2
        threshold = initial_mem_usage * 1.02

        byte_size = 4
        i = 0
        while i < 100000:
            if i % 5000 == 0:
                print(
                    f"[iter: {i:<8}] Memory Usage:",
                    process.memory_info().rss / 1024**2,
                    "MiB",
                )

            shm_handle = shm.create_shared_memory_region(
                "shmtest", "/shmtest", byte_size
            )
            shm.destroy_shared_memory_region(shm_handle)
            i += 1
        final_mem_usage = process.memory_info().rss / 1024**2
        self.assertTrue(
            (initial_mem_usage <= final_mem_usage <= threshold),
            "client memory usage is increasing",
        )


def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)


class TestSharedMemoryUnregister(SystemSharedMemoryTestBase):
    def _create_request_data(self):
        self.triton_client.unregister_system_shared_memory()
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
            second_client.unregister_system_shared_memory()

        self._test_shm_found()

    def _test_shm_not_found(self):
        second_client = httpclient.InferenceServerClient("localhost:8000", verbose=True)

        for shm_name in self.shm_names:
            with self.assertRaises(utils.InferenceServerException) as ex:
                second_client.get_system_shared_memory_status(shm_name)
                self.assertIn(
                    f"Unable to find system shared memory region: '{shm_name}'",
                    str(ex.exception),
                )

    def _test_shm_found(self):
        second_client = httpclient.InferenceServerClient("localhost:8000", verbose=True)

        status = second_client.get_system_shared_memory_status()
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

        self.triton_client.unregister_system_shared_memory()
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

        self.triton_client.unregister_system_shared_memory()
        self._test_shm_not_found()


if __name__ == "__main__":
    unittest.main()
