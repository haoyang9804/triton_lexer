import unittest

import numpy


import torch
import tritonclient.utils as utils
import tritonclient.utils.cuda_shared_memory as cudashm


class DLPackTest(unittest.TestCase):

    def test_from_gpu(self):

        gpu_tensor = torch.ones(4, 4).cuda(0)
        byte_size = 64
        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        cudashm.set_shared_memory_region_from_dlpack(shm_handle, [gpu_tensor])

        smt = cudashm.as_shared_memory_tensor(shm_handle, "FP32", [4, 4])
        generated_torch_tensor = torch.from_dlpack(smt)
        self.assertTrue(torch.allclose(gpu_tensor, generated_torch_tensor))

        cudashm.destroy_shared_memory_region(shm_handle)

    def test_from_cpu(self):

        cpu_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        byte_size = 64
        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        cudashm.set_shared_memory_region_from_dlpack(shm_handle, [cpu_tensor])

        smt = cudashm.as_shared_memory_tensor(shm_handle, "FP32", [4, 4])
        generated_torch_tensor = torch.from_dlpack(smt)

        self.assertTrue(
            numpy.allclose(cpu_tensor, numpy.from_dlpack(generated_torch_tensor.cpu()))
        )

        cudashm.destroy_shared_memory_region(shm_handle)


class NumpyTest(unittest.TestCase):

    def test_from_numpy(self):

        cpu_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        byte_size = 64
        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        cudashm.set_shared_memory_region(shm_handle, [cpu_tensor])

        smt = cudashm.as_shared_memory_tensor(shm_handle, "FP32", [4, 4])
        generated_torch_tensor = torch.from_dlpack(smt)

        self.assertTrue(
            numpy.allclose(cpu_tensor, numpy.from_dlpack(generated_torch_tensor.cpu()))
        )

        cudashm.destroy_shared_memory_region(shm_handle)

    def test_to_numpy(self):

        cpu_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        byte_size = 64
        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        cudashm.set_shared_memory_region(shm_handle, [cpu_tensor])

        generated_tensor = cudashm.get_contents_as_numpy(
            shm_handle, numpy.float32, [4, 4]
        )

        self.assertTrue(numpy.allclose(cpu_tensor, generated_tensor))

        cudashm.destroy_shared_memory_region(shm_handle)

    def test_numpy_bytes(self):
        int_tensor = numpy.arange(start=0, stop=16, dtype=numpy.int32)
        bytes_tensor = numpy.array(
            [str(x).encode("utf-8") for x in int_tensor.flatten()], dtype=object
        )
        bytes_tensor = bytes_tensor.reshape(int_tensor.shape)
        bytes_tensor_serialized = utils.serialize_byte_tensor(bytes_tensor)
        byte_size = utils.serialized_byte_size(bytes_tensor_serialized)

        shm_handle = cudashm.create_shared_memory_region("cudashm_data", byte_size, 0)

        cudashm.set_shared_memory_region(shm_handle, [bytes_tensor_serialized])

        generated_tensor = cudashm.get_contents_as_numpy(
            shm_handle,
            numpy.object_,
            [
                16,
            ],
        )

        self.assertTrue(numpy.array_equal(bytes_tensor, generated_tensor))

        cudashm.destroy_shared_memory_region(shm_handle)


if __name__ == "__main__":
    unittest.main()
