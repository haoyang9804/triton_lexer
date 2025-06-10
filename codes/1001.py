import sys

sys.path.append("../common")

import os
import unittest

import numpy as np
import requests as httpreq
import shm_util
import tritonclient.http as httpclient
from tritonclient.utils import *


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

_test_jetson = bool(int(os.environ.get("TEST_JETSON", 0)))
_test_windows = bool(int(os.environ.get("TEST_WINDOWS", 0)))


class PythonTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def _infer_help(self, model_name, shape, data_type):
        with httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000") as client:
            input_data_0 = np.array(np.random.randn(*shape), dtype=data_type)
            inputs = [
                httpclient.InferInput(
                    "INPUT0", shape, np_to_triton_dtype(input_data_0.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data_0)

            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("OUTPUT0")
            self.assertTrue(np.all(input_data_0 == output0))

    def _create_cuda_region(self, client, size, name):
        import tritonclient.utils.cuda_shared_memory as cuda_shared_memory

        shm0_handle = cuda_shared_memory.create_shared_memory_region(
            name, byte_size=size, device_id=0
        )
        client.register_cuda_shared_memory(
            name, cuda_shared_memory.get_raw_handle(shm0_handle), 0, size
        )
        return shm0_handle

    def _optional_input_infer(self, model_name, has_input0, has_input1):
        with httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000") as client:
            shape = (1,)
            if has_input0:
                input0_numpy = np.random.randint(0, 100, size=shape, dtype=np.int32)
            else:

                input0_numpy = np.array([5], dtype=np.int32)

            if has_input1:
                input1_numpy = np.random.randint(0, 100, size=shape, dtype=np.int32)
            else:

                input1_numpy = np.array([5], dtype=np.int32)

            inputs = []
            if has_input0:
                inputs.append(
                    httpclient.InferInput(
                        "INPUT0", shape, np_to_triton_dtype(input0_numpy.dtype)
                    )
                )
                inputs[-1].set_data_from_numpy(input0_numpy)

            if has_input1:
                inputs.append(
                    httpclient.InferInput(
                        "INPUT1", shape, np_to_triton_dtype(input1_numpy.dtype)
                    )
                )
                inputs[-1].set_data_from_numpy(input1_numpy)

            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("OUTPUT0")
            self.assertIsNotNone(output0, "OUTPUT0 was not found.")

            output1 = result.as_numpy("OUTPUT1")
            self.assertIsNotNone(output1, "OUTPUT1 was not found.")

            expected_output0 = input0_numpy + input1_numpy
            expected_output1 = input0_numpy - input1_numpy
            np.testing.assert_equal(
                output0, expected_output0, "OUTPUT0 doesn't match expected OUTPUT0"
            )
            np.testing.assert_equal(
                output1, expected_output1, "OUTPUT1 doesn't match expected OUTPUT1"
            )

    def test_growth_error(self):

        if not _test_windows:

            total_byte_size = 2 * 1024 * 1024
            shape = [total_byte_size]
            model_name = "identity_uint8_nobatch"
            dtype = np.uint8
            with self._shm_leak_detector.Probe() as shm_probe:
                self._infer_help(model_name, shape, dtype)

            total_byte_size = 1024 * 1024 * 1024
            shape = [total_byte_size]
            with self.assertRaises(InferenceServerException) as ex:
                self._infer_help(model_name, shape, dtype)
            self.assertIn(
                "Failed to increase the shared memory pool size", str(ex.exception)
            )

            total_byte_size = 512 * 1024 * 1024
            shape = [total_byte_size]
            with self.assertRaises(InferenceServerException) as ex:
                self._infer_help(model_name, shape, dtype)
            self.assertIn(
                "Failed to increase the shared memory pool size", str(ex.exception)
            )

            total_byte_size = 2 * 1024 * 1024
            shape = [total_byte_size]
            with self._shm_leak_detector.Probe() as shm_probe:
                self._infer_help(model_name, shape, dtype)

    if not _test_jetson and not _test_windows:

        def test_gpu_tensor_error(self):
            import tritonclient.utils.cuda_shared_memory as cuda_shared_memory

            model_name = "identity_bool"
            with self._shm_leak_detector.Probe() as shm_probe:
                with httpclient.InferenceServerClient(
                    f"{_tritonserver_ipaddr}:8000"
                ) as client:
                    input_data = np.array([[True] * 1000], dtype=bool)
                    inputs = [
                        httpclient.InferInput(
                            "INPUT0",
                            input_data.shape,
                            np_to_triton_dtype(input_data.dtype),
                        )
                    ]
                    inputs[0].set_data_from_numpy(input_data)

                    requested_outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

                    client.unregister_cuda_shared_memory()
                    shm0_handle = self._create_cuda_region(client, 1, "output0_data")

                    requested_outputs[0].set_shared_memory("output0_data", 1)
                    with self.assertRaises(InferenceServerException) as ex:
                        client.infer(model_name, inputs, outputs=requested_outputs)
                    self.assertIn(
                        "should be at least 1000 bytes to hold the results",
                        str(ex.exception),
                    )
                    client.unregister_cuda_shared_memory()
                    cuda_shared_memory.destroy_shared_memory_region(shm0_handle)

        def test_dlpack_tensor_error(self):
            import tritonclient.utils.cuda_shared_memory as cuda_shared_memory

            model_name = "dlpack_identity"
            with self._shm_leak_detector.Probe() as shm_probe:
                with httpclient.InferenceServerClient(
                    f"{_tritonserver_ipaddr}:8000"
                ) as client:
                    input_data = np.array([[1] * 1000], dtype=np.float32)
                    inputs = [
                        httpclient.InferInput(
                            "INPUT0",
                            input_data.shape,
                            np_to_triton_dtype(input_data.dtype),
                        )
                    ]

                    requested_outputs = [httpclient.InferRequestedOutput("OUTPUT0")]
                    input_data_size = input_data.itemsize * input_data.size
                    client.unregister_cuda_shared_memory()
                    input_region = self._create_cuda_region(
                        client, input_data_size, "input0_data"
                    )
                    inputs[0].set_shared_memory("input0_data", input_data_size)
                    cuda_shared_memory.set_shared_memory_region(
                        input_region, [input_data]
                    )

                    shm0_handle = self._create_cuda_region(client, 1, "output0_data")
                    requested_outputs[0].set_shared_memory("output0_data", 1)

                    with self.assertRaises(InferenceServerException) as ex:
                        client.infer(model_name, inputs, outputs=requested_outputs)
                    self.assertIn(
                        "should be at least 4000 bytes to hold the results",
                        str(ex.exception),
                    )
                    client.unregister_cuda_shared_memory()
                    cuda_shared_memory.destroy_shared_memory_region(shm0_handle)

    def test_async_infer(self):
        model_name = "identity_uint8"
        request_parallelism = 4
        shape = [2, 2]

        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000", concurrency=request_parallelism
            ) as client:
                input_datas = []
                requests = []
                for i in range(request_parallelism):
                    input_data = (16384 * np.random.randn(*shape)).astype(np.uint8)
                    input_datas.append(input_data)
                    inputs = [
                        httpclient.InferInput(
                            "INPUT0",
                            input_data.shape,
                            np_to_triton_dtype(input_data.dtype),
                        )
                    ]
                    inputs[0].set_data_from_numpy(input_data)
                    requests.append(client.async_infer(model_name, inputs))

                for i in range(request_parallelism):

                    results = requests[i].get_result()

                    output_data = results.as_numpy("OUTPUT0")
                    self.assertIsNotNone(output_data, "error: expected 'OUTPUT0'")
                    self.assertTrue(
                        np.array_equal(output_data, input_datas[i]),
                        "error: expected output {} to match input {}".format(
                            output_data, input_datas[i]
                        ),
                    )

                stats = client.get_inference_statistics(model_name)
                test_cond = (len(stats["model_stats"]) != 1) or (
                    stats["model_stats"][0]["name"] != model_name
                )
                self.assertFalse(
                    test_cond, "error: expected statistics for {}".format(model_name)
                )

                stat = stats["model_stats"][0]
                self.assertFalse(
                    (stat["inference_count"] != 8) or (stat["execution_count"] != 1),
                    "error: expected execution_count == 1 and inference_count == 8, got {} and {}".format(
                        stat["execution_count"], stat["inference_count"]
                    ),
                )
                batch_stat = stat["batch_stats"][0]
                self.assertFalse(
                    batch_stat["batch_size"] != 8,
                    f"error: expected batch_size == 8, got {batch_stat['batch_size']}",
                )

                metrics = httpreq.get(f"http://{_tritonserver_ipaddr}:8002/metrics")
                print(metrics.text)

                success_str = (
                    'nv_inference_request_success{model="identity_uint8",version="1"}'
                )
                infer_count_str = (
                    'nv_inference_count{model="identity_uint8",version="1"}'
                )
                infer_exec_str = (
                    'nv_inference_exec_count{model="identity_uint8",version="1"}'
                )

                success_val = None
                infer_count_val = None
                infer_exec_val = None
                for line in metrics.text.splitlines():
                    if line.startswith(success_str):
                        success_val = float(line[len(success_str) :])
                    if line.startswith(infer_count_str):
                        infer_count_val = float(line[len(infer_count_str) :])
                    if line.startswith(infer_exec_str):
                        infer_exec_val = float(line[len(infer_exec_str) :])

                self.assertFalse(
                    success_val != 4,
                    "error: expected metric {} == 4, got {}".format(
                        success_str, success_val
                    ),
                )
                self.assertFalse(
                    infer_count_val != 8,
                    "error: expected metric {} == 8, got {}".format(
                        infer_count_str, infer_count_val
                    ),
                )
                self.assertFalse(
                    infer_exec_val != 1,
                    "error: expected metric {} == 1, got {}".format(
                        infer_exec_str, infer_exec_val
                    ),
                )

    def test_bool(self):
        model_name = "identity_bool"
        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000"
            ) as client:
                input_data = np.array([[True, False, True]], dtype=bool)
                inputs = [
                    httpclient.InferInput(
                        "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("OUTPUT0")
                self.assertIsNotNone(output0)
                self.assertTrue(np.all(output0 == input_data))

    def test_bf16(self):
        model_name = "identity_bf16"
        shape = [2, 2]
        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000"
            ) as client:

                np_input = np.ones(shape, dtype=np.float32)
                inputs = [
                    httpclient.InferInput(
                        "INPUT0", np_input.shape, "BF16"
                    ).set_data_from_numpy(np_input)
                ]
                result = client.infer(model_name, inputs)

                response = result.get_response()
                triton_output = response["outputs"][0]
                triton_dtype = triton_output["datatype"]
                self.assertEqual(triton_dtype, "BF16")

                np_output = result.as_numpy("OUTPUT0")
                self.assertIsNotNone(np_output)

                self.assertEqual(np_output.dtype, np.float32)
                self.assertTrue(np.allclose(np_output, np_input))

    def test_infer_pytorch(self):

        if not _test_windows:
            model_name = "pytorch_fp32_fp32"
            shape = [1, 1, 28, 28]
            with self._shm_leak_detector.Probe() as shm_probe:
                with httpclient.InferenceServerClient(
                    f"{_tritonserver_ipaddr}:8000"
                ) as client:
                    input_data = np.zeros(shape, dtype=np.float32)
                    inputs = [
                        httpclient.InferInput(
                            "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                        )
                    ]
                    inputs[0].set_data_from_numpy(input_data)
                    result = client.infer(model_name, inputs)
                    output_data = result.as_numpy("OUT")
                    self.assertIsNotNone(output_data, "error: expected 'OUT'")

                    expected_result = [
                        -2.2377274,
                        -2.3976364,
                        -2.2464046,
                        -2.2790744,
                        -2.3828976,
                        -2.2940576,
                        -2.2928185,
                        -2.340665,
                        -2.275219,
                        -2.292135,
                    ]
                    self.assertTrue(
                        np.allclose(output_data[0], expected_result),
                        "Inference result is not correct",
                    )

    def test_init_args(self):
        model_name = "init_args"
        shape = [2, 2]
        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000"
            ) as client:
                input_data = np.zeros(shape, dtype=np.float32)
                inputs = [
                    httpclient.InferInput(
                        "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_data)
                result = client.infer(model_name, inputs)

                self.assertTrue(
                    result.as_numpy("OUT") == 7,
                    "Number of keys in the init args is not correct",
                )

    def test_unicode(self):
        model_name = "string"
        shape = [1]

        for i in range(2):
            with self._shm_leak_detector.Probe() as shm_probe:
                with httpclient.InferenceServerClient(
                    f"{_tritonserver_ipaddr}:8000"
                ) as client:
                    utf8 = "😀"
                    input_data = np.array(
                        [bytes(utf8, encoding="utf-8")], dtype=np.bytes_
                    )
                    inputs = [
                        httpclient.InferInput(
                            "INPUT0", shape, np_to_triton_dtype(input_data.dtype)
                        )
                    ]
                    inputs[0].set_data_from_numpy(input_data)
                    result = client.infer(model_name, inputs)
                    output0 = result.as_numpy("OUTPUT0")
                    self.assertIsNotNone(output0)
                    self.assertEqual(output0[0], input_data)

    def test_optional_input(self):
        model_name = "optional"

        with self._shm_leak_detector.Probe() as shm_probe:
            for has_input0 in [True, False]:
                for has_input1 in [True, False]:
                    self._optional_input_infer(model_name, has_input0, has_input1)

    def test_string(self):
        model_name = "string_fixed"
        shape = [1]

        for i in range(4):
            with self._shm_leak_detector.Probe() as shm_probe:
                with httpclient.InferenceServerClient(
                    f"{_tritonserver_ipaddr}:8000"
                ) as client:
                    input_data = np.array(["123456"], dtype=np.object_)
                    inputs = [
                        httpclient.InferInput(
                            "INPUT0", shape, np_to_triton_dtype(input_data.dtype)
                        )
                    ]
                    inputs[0].set_data_from_numpy(input_data)
                    result = client.infer(model_name, inputs)
                    output0 = result.as_numpy("OUTPUT0")
                    self.assertIsNotNone(output0)

                    if i % 2 == 0:
                        self.assertEqual(output0[0], input_data.astype(np.bytes_))
                    else:
                        self.assertEqual(output0.size, 0)

    def test_non_contiguous(self):
        model_name = "non_contiguous"
        shape = [2, 10, 11, 6, 5]
        new_shape = [10, 2, 6, 5, 11]
        shape_reorder = [1, 0, 4, 2, 3]
        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000"
            ) as client:
                input_numpy = np.random.rand(*shape)
                input_numpy = input_numpy.astype(np.float32)
                inputs = [
                    httpclient.InferInput(
                        "INPUT0", shape, np_to_triton_dtype(input_numpy.dtype)
                    )
                ]
                inputs[0].set_data_from_numpy(input_numpy)
                result = client.infer(model_name, inputs)
                output0 = input_numpy.reshape(new_shape)

                output1 = input_numpy.T
                output2 = np.transpose(input_numpy, shape_reorder)

                self.assertTrue(np.all(output0 == result.as_numpy("OUTPUT0")))
                self.assertTrue(np.all(output1 == result.as_numpy("OUTPUT1")))
                self.assertTrue(np.all(output2 == result.as_numpy("OUTPUT2")))


if __name__ == "__main__":
    unittest.main()
