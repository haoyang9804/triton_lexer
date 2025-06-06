import unittest

import numpy
import tritonclient.utils as utils
import tritonclient.utils.shared_memory as shm


class SharedMemoryTest(unittest.TestCase):

    def setUp(self):
        self.shm_handles = []

    def tearDown(self):
        for shm_handle in self.shm_handles:
            shm.destroy_shared_memory_region(shm_handle)

    def test_lifecycle(self):
        cpu_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        byte_size = 64
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", byte_size)
        )

        self.assertEqual(len(shm.mapped_shared_memory_regions()), 1)

        shm.set_shared_memory_region(self.shm_handles[0], [cpu_tensor])
        shm_tensor = shm.get_contents_as_numpy(
            self.shm_handles[0], numpy.float32, [4, 4]
        )

        self.assertTrue(numpy.allclose(cpu_tensor, shm_tensor))

        shm.destroy_shared_memory_region(self.shm_handles.pop(0))

    def test_invalid_create_shm(self):

        with self.assertRaisesRegex(
            shm.SharedMemoryException, "unable to create the shared memory region"
        ):
            self.shm_handles.append(
                shm.create_shared_memory_region("dummy_data", "/dummy_data", -1)
            )

    def test_set_region_offset(self):
        large_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        large_size = 64
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", large_size)
        )
        shm.set_shared_memory_region(self.shm_handles[0], [large_tensor])
        small_tensor = numpy.zeros([2, 4], dtype=numpy.float32)
        small_size = 32
        shm.set_shared_memory_region(
            self.shm_handles[0], [small_tensor], offset=large_size - small_size
        )
        shm_tensor = shm.get_contents_as_numpy(
            self.shm_handles[0], numpy.float32, [2, 4], offset=large_size - small_size
        )

        self.assertTrue(numpy.allclose(small_tensor, shm_tensor))

    def test_set_region_oversize(self):
        large_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        small_size = 32
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", small_size)
        )
        with self.assertRaisesRegex(
            shm.SharedMemoryException, "unable to set the shared memory region"
        ):
            shm.set_shared_memory_region(self.shm_handles[0], [large_tensor])

    def test_duplicate_key(self):

        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 32)
        )
        with self.assertRaisesRegex(
            shm.SharedMemoryException,
            "unable to create the shared memory region",
        ):
            self.shm_handles.append(
                shm.create_shared_memory_region(
                    "shm_name", "shm_key", 32, create_only=True
                )
            )

        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 64)
        )

        self.assertEqual(len(shm.mapped_shared_memory_regions()), 1)

        large_tensor = numpy.ones([4, 4], dtype=numpy.float32)
        with self.assertRaisesRegex(
            shm.SharedMemoryException, "unable to set the shared memory region"
        ):
            shm.set_shared_memory_region(self.shm_handles[-1], [large_tensor])

    def test_destroy_duplicate(self):

        self.assertEqual(len(shm.mapped_shared_memory_regions()), 0)
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 64)
        )
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 32)
        )
        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", 32)
        )
        self.assertEqual(len(shm.mapped_shared_memory_regions()), 1)

        shm.destroy_shared_memory_region(self.shm_handles.pop(0))
        shm.destroy_shared_memory_region(self.shm_handles.pop(0))
        self.assertEqual(len(shm.mapped_shared_memory_regions()), 1)

        shm.destroy_shared_memory_region(self.shm_handles.pop(0))
        self.assertEqual(len(shm.mapped_shared_memory_regions()), 0)

    def test_numpy_bytes(self):
        int_tensor = numpy.arange(start=0, stop=16, dtype=numpy.int32)
        bytes_tensor = numpy.array(
            [str(x).encode("utf-8") for x in int_tensor.flatten()], dtype=object
        )
        bytes_tensor = bytes_tensor.reshape(int_tensor.shape)
        bytes_tensor_serialized = utils.serialize_byte_tensor(bytes_tensor)
        byte_size = utils.serialized_byte_size(bytes_tensor_serialized)

        self.shm_handles.append(
            shm.create_shared_memory_region("shm_name", "shm_key", byte_size)
        )

        shm.set_shared_memory_region(self.shm_handles[0], [bytes_tensor_serialized])

        shm_tensor = shm.get_contents_as_numpy(
            self.shm_handles[0],
            numpy.object_,
            [
                16,
            ],
        )

        self.assertTrue(numpy.array_equal(bytes_tensor, shm_tensor))


if __name__ == "__main__":
    unittest.main()
