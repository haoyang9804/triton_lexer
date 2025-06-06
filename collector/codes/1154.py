import unittest

import cupy as cp
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


class PBTensorTest(unittest.TestCase):
    def test_pytorch_dlpack(self):

        pytorch_dtypes = [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ]

        for pytorch_dtype in pytorch_dtypes:
            pytorch_tensor = torch.ones([100], dtype=pytorch_dtype)
            dlpack_tensor = to_dlpack(pytorch_tensor)
            pb_tensor = pb_utils.Tensor.from_dlpack("test_tensor", dlpack_tensor)
            self.assertTrue(
                np.array_equal(pb_tensor.as_numpy(), pytorch_tensor.numpy())
            )

            pytorch_tensor_dlpack = from_dlpack(pb_tensor.to_dlpack())
            self.assertTrue(torch.equal(pytorch_tensor_dlpack, pytorch_tensor))

            self.assertEqual(pytorch_tensor.type(), pytorch_tensor_dlpack.type())

            pb_tensor_upgraded = pb_utils.Tensor.from_dlpack(
                "test_tensor", pytorch_tensor
            )
            self.assertTrue(
                np.array_equal(pb_tensor_upgraded.as_numpy(), pytorch_tensor.numpy())
            )

            pytorch_tensor_dlpack = from_dlpack(pb_tensor_upgraded)
            self.assertTrue(torch.equal(pytorch_tensor_dlpack, pytorch_tensor))

            self.assertEqual(pytorch_tensor.type(), pytorch_tensor_dlpack.type())

    def test_non_contiguous_error(self):
        pytorch_tensor = torch.rand([20, 30], dtype=torch.float16)

        pytorch_tensor = torch.transpose(pytorch_tensor, 0, 1)

        with self.assertRaises(Exception) as e:
            pb_utils.Tensor.from_dlpack("test_tensor", to_dlpack(pytorch_tensor))
        self.assertTrue(
            str(e.exception)
            == "DLPack tensor is not contiguous. Only contiguous DLPack tensors that are stored in C-Order are supported."
        )

    def test_dlpack_string_tensor(self):
        np_object = np.array(["An Example String"], dtype=np.object_)
        pb_tensor = pb_utils.Tensor("test_tensor", np_object)

        with self.assertRaises(Exception) as e:
            pb_tensor.to_dlpack()

        self.assertTrue(
            str(e.exception) == "DLPack does not have support for string tensors."
        )

    def test_dlpack_gpu_tensors(self):

        pytorch_dtypes = [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ]

        for pytorch_dtype in pytorch_dtypes:
            pytorch_tensor = torch.ones([100], dtype=pytorch_dtype, device="cuda")
            dlpack_tensor = to_dlpack(pytorch_tensor)
            pb_tensor = pb_utils.Tensor.from_dlpack("test_tensor", dlpack_tensor)

            pytorch_tensor_dlpack = from_dlpack(pb_tensor.to_dlpack())
            self.assertTrue(torch.equal(pytorch_tensor_dlpack, pytorch_tensor))
            self.assertEqual(pytorch_tensor.type(), pytorch_tensor_dlpack.type())

            pb_tensor = pb_utils.Tensor.from_dlpack("test_tensor", pytorch_tensor)
            pytorch_tensor_dlpack = from_dlpack(pb_tensor)
            self.assertTrue(torch.equal(pytorch_tensor_dlpack, pytorch_tensor))
            self.assertEqual(pytorch_tensor.type(), pytorch_tensor_dlpack.type())

    def test_dlpack_gpu_numpy(self):

        pytorch_tensor = torch.rand([100], dtype=torch.float16, device="cuda") * 100
        pb_tensor = pb_utils.Tensor.from_dlpack("tensor", to_dlpack(pytorch_tensor))

        self.assertFalse(pb_tensor.is_cpu())
        self.assertTrue(pytorch_tensor.is_cuda)
        self.assertEqual(
            pb_tensor.__dlpack_device__(), pytorch_tensor.__dlpack_device__()
        )

        with self.assertRaises(Exception) as e:
            pb_tensor.as_numpy()
        self.assertTrue(
            str(e.exception)
            == "Tensor is stored in GPU and cannot be converted to NumPy."
        )

    def test_dlpack_cpu_numpy(self):

        pytorch_tensor = torch.rand([100], dtype=torch.float16, device="cpu") * 100
        pb_tensor = pb_utils.Tensor.from_dlpack("tensor", pytorch_tensor)
        numpy_tensor_dlpack = np.from_dlpack(pb_tensor)
        self.assertTrue(np.array_equal(numpy_tensor_dlpack, pytorch_tensor.numpy()))

        self.assertTrue(pb_tensor.is_cpu())
        self.assertFalse(pytorch_tensor.is_cuda)
        self.assertEqual(
            pb_tensor.__dlpack_device__(), pytorch_tensor.__dlpack_device__()
        )

    def test_bool_datatype(self):

        bool_array = np.asarray([False, True])
        bool_tensor = pb_utils.Tensor("tensor", bool_array)
        bool_tensor_dlpack = pb_utils.Tensor.from_dlpack("tensor", bool_tensor)
        self.assertTrue(np.array_equal(bool_array, bool_tensor_dlpack.as_numpy()))

    def test_cuda_multi_stream(self):

        size = 5000
        pytorch_tensor_1 = torch.tensor([0, 0, 0, 0], device="cuda")
        pytorch_tensor_2 = torch.tensor([0, 0, 0, 0], device="cuda")
        expected_output = torch.tensor([2, 2, 2, 2], device="cuda")
        s1 = torch.cuda.Stream()
        with torch.cuda.stream(s1):
            matrix_a = torch.randn(size, size, device="cuda")
            res = torch.matmul(matrix_a, matrix_a)
            for _ in range(1000):
                res = torch.matmul(res, matrix_a)
            pytorch_tensor_1 += torch.tensor([2, 2, 2, 2], device="cuda")
            pytorch_tensor_2 += torch.tensor([2, 2, 2, 2], device="cuda")

        pb_tensor_1 = pb_utils.Tensor.from_dlpack("tensor", pytorch_tensor_1)
        pb_tensor_2 = pb_utils.Tensor.from_dlpack("tensor", to_dlpack(pytorch_tensor_2))
        pytorch_tensor_dlpack = from_dlpack(pb_tensor_1)
        self.assertTrue(torch.equal(pytorch_tensor_dlpack, expected_output))
        pytorch_tensor_dlpack = from_dlpack(pb_tensor_2)
        self.assertTrue(torch.equal(pytorch_tensor_dlpack, expected_output))

    def test_cuda_non_blocking_multi_stream(self):

        size = 5000
        cupy_tensor = cp.array([0, 0, 0, 0])
        expected_output = cp.array([2, 2, 2, 2])
        non_blocking_stream = cp.cuda.Stream(non_blocking=True)
        with non_blocking_stream:
            matrix_a = cp.random.rand(size, size)
            res = cp.matmul(matrix_a, matrix_a)
            for _ in range(1000):
                res = cp.matmul(res, matrix_a)
            cupy_tensor += cp.array([2, 2, 2, 2])

        pb_tensor = pb_utils.Tensor.from_dlpack("tensor", cupy_tensor)

        self.assertTrue(non_blocking_stream.done)
        cupy_tensor_dlpack = cp.from_dlpack(pb_tensor)
        self.assertTrue(cp.array_equal(cupy_tensor_dlpack, expected_output))
        self.assertFalse(pb_tensor.is_cpu())
        self.assertEqual(pb_tensor.__dlpack_device__(), cupy_tensor.__dlpack_device__())

    def test_cuda_multi_gpu(self):

        size = 5000

        expected_dlpack_device = (2, 1)
        with cp.cuda.Device(1):
            expected_output = cp.array([2, 2, 2, 2])
            cupy_tensor = cp.array([0, 0, 0, 0])
            matrix_a = cp.random.rand(size, size)
            res = cp.matmul(matrix_a, matrix_a)
            for _ in range(1000):
                res = cp.matmul(res, matrix_a)
            cupy_tensor += cp.array([2, 2, 2, 2])
        with cp.cuda.Device(0):
            pb_tensor = pb_utils.Tensor.from_dlpack("tensor", cupy_tensor)
            with cp.cuda.Device(1):

                self.assertTrue(cp.cuda.Stream(null=True).done)
            cupy_tensor_dlpack = cp.from_dlpack(pb_tensor)

        with cp.cuda.Device(1):
            self.assertTrue(cp.array_equal(cupy_tensor_dlpack, expected_output))

        self.assertFalse(pb_tensor.is_cpu())
        self.assertEqual(pb_tensor.__dlpack_device__(), expected_dlpack_device)
        self.assertEqual(pb_tensor.__dlpack_device__(), cupy_tensor.__dlpack_device__())

    def test_cuda_blocking_stream_multi_gpu(self):

        size = 5000

        expected_dlpack_device = (2, 1)
        with cp.cuda.Device(1):
            expected_output = cp.array([2, 2, 2, 2])
            blocking_stream = cp.cuda.Stream(non_blocking=False)
            with blocking_stream:
                cupy_tensor = cp.array([0, 0, 0, 0])
                matrix_a = cp.random.rand(size, size)
                res = cp.matmul(matrix_a, matrix_a)
                for _ in range(1000):
                    res = cp.matmul(res, matrix_a)
                cupy_tensor += cp.array([2, 2, 2, 2])
        with cp.cuda.Device(0):
            pb_tensor = pb_utils.Tensor.from_dlpack("tensor", cupy_tensor)
            with cp.cuda.Device(1):

                self.assertTrue(blocking_stream.done)
            cupy_tensor_dlpack = cp.from_dlpack(pb_tensor)

        with cp.cuda.Device(1):
            self.assertTrue(cp.array_equal(cupy_tensor_dlpack, expected_output))

        self.assertFalse(pb_tensor.is_cpu())
        self.assertEqual(pb_tensor.__dlpack_device__(), expected_dlpack_device)
        self.assertEqual(pb_tensor.__dlpack_device__(), cupy_tensor.__dlpack_device__())

    def test_cuda_non_blocking_stream_multi_gpu(self):

        size = 5000

        expected_dlpack_device = (2, 2)
        with cp.cuda.Device(2):
            expected_output = cp.array([2, 2, 2, 2])
            non_blocking_stream = cp.cuda.Stream(non_blocking=True)
            with non_blocking_stream:
                cupy_tensor = cp.array([0, 0, 0, 0])
                matrix_a = cp.random.rand(size, size)
                res = cp.matmul(matrix_a, matrix_a)
                for _ in range(1000):
                    res = cp.matmul(res, matrix_a)
                cupy_tensor += cp.array([2, 2, 2, 2])
        with cp.cuda.Device(0):
            pb_tensor = pb_utils.Tensor.from_dlpack("tensor", cupy_tensor)
            with cp.cuda.Device(2):

                self.assertTrue(non_blocking_stream.done)
            cupy_tensor_dlpack = cp.from_dlpack(pb_tensor)

        with cp.cuda.Device(2):
            self.assertTrue(cp.array_equal(cupy_tensor_dlpack, expected_output))

        self.assertFalse(pb_tensor.is_cpu())
        self.assertEqual(pb_tensor.__dlpack_device__(), expected_dlpack_device)
        self.assertEqual(pb_tensor.__dlpack_device__(), cupy_tensor.__dlpack_device__())


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for _ in requests:

            test = unittest.main("model", exit=False)
            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor(
                            "OUTPUT0",
                            np.array([test.result.wasSuccessful()], dtype=np.float16),
                        )
                    ]
                )
            )
        return responses
