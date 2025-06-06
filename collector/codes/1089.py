import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as tritongrpcclient
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import cuda_shared_memory as cudashm


class QueryTest(tu.TestResultCollector):
    def test_http(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT", [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        try:
            triton_client.infer(model_name="query", inputs=inputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 CPU 0" in ex.message())
            self.assertTrue("OUTPUT1 CPU 0" in ex.message())

    def test_http_shared_memory(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT", [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()
        shm_op0_handle = cudashm.create_shared_memory_region("output0_data", 4, 0)
        shm_op1_handle = cudashm.create_shared_memory_region("output1_data", 4, 0)
        triton_client.register_cuda_shared_memory(
            "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, 4
        )
        triton_client.register_cuda_shared_memory(
            "output1_data", cudashm.get_raw_handle(shm_op1_handle), 0, 4
        )
        outputs = []
        outputs.append(
            tritonhttpclient.InferRequestedOutput("OUTPUT0", binary_data=True)
        )
        outputs[-1].set_shared_memory("output0_data", 4)

        outputs.append(
            tritonhttpclient.InferRequestedOutput("OUTPUT1", binary_data=True)
        )
        outputs[-1].set_shared_memory("output1_data", 4)

        try:
            triton_client.infer(model_name="query", inputs=inputs, outputs=outputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 GPU 0" in ex.message())
            self.assertTrue("OUTPUT1 GPU 0" in ex.message())

        cudashm.destroy_shared_memory_region(shm_op0_handle)
        cudashm.destroy_shared_memory_region(shm_op1_handle)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()

    def test_http_out_of_shared_memory(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT", [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()
        shm_op0_handle = cudashm.create_shared_memory_region("output0_data", 1, 0)
        shm_op1_handle = cudashm.create_shared_memory_region("output1_data", 1, 0)
        triton_client.register_cuda_shared_memory(
            "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, 1
        )
        triton_client.register_cuda_shared_memory(
            "output1_data", cudashm.get_raw_handle(shm_op1_handle), 0, 1
        )
        outputs = []
        outputs.append(
            tritonhttpclient.InferRequestedOutput("OUTPUT0", binary_data=True)
        )
        outputs[-1].set_shared_memory("output0_data", 1)

        outputs.append(
            tritonhttpclient.InferRequestedOutput("OUTPUT1", binary_data=True)
        )
        outputs[-1].set_shared_memory("output1_data", 1)

        try:
            triton_client.infer(model_name="query", inputs=inputs, outputs=outputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 CPU 0" in ex.message())
            self.assertTrue("OUTPUT1 CPU 0" in ex.message())

        cudashm.destroy_shared_memory_region(shm_op0_handle)
        cudashm.destroy_shared_memory_region(shm_op1_handle)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()

    def test_grpc(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT", [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        try:
            triton_client.infer(model_name="query", inputs=inputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 CPU 0" in ex.message())
            self.assertTrue("OUTPUT1 CPU 0" in ex.message())

    def test_grpc_shared_memory(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT", [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()
        shm_op0_handle = cudashm.create_shared_memory_region("output0_data", 4, 0)
        shm_op1_handle = cudashm.create_shared_memory_region("output1_data", 4, 0)
        triton_client.register_cuda_shared_memory(
            "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, 4
        )
        triton_client.register_cuda_shared_memory(
            "output1_data", cudashm.get_raw_handle(shm_op1_handle), 0, 4
        )
        outputs = []
        outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT0"))
        outputs[-1].set_shared_memory("output0_data", 4)

        outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT1"))
        outputs[-1].set_shared_memory("output1_data", 4)

        try:
            triton_client.infer(model_name="query", inputs=inputs, outputs=outputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 GPU 0" in ex.message())
            self.assertTrue("OUTPUT1 GPU 0" in ex.message())

        cudashm.destroy_shared_memory_region(shm_op0_handle)
        cudashm.destroy_shared_memory_region(shm_op1_handle)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()

    def test_grpc_out_of_shared_memory(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT", [1], "UINT8"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.uint8))

        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()
        shm_op0_handle = cudashm.create_shared_memory_region("output0_data", 1, 0)
        shm_op1_handle = cudashm.create_shared_memory_region("output1_data", 1, 0)
        triton_client.register_cuda_shared_memory(
            "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, 1
        )
        triton_client.register_cuda_shared_memory(
            "output1_data", cudashm.get_raw_handle(shm_op1_handle), 0, 1
        )
        outputs = []
        outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT0"))
        outputs[-1].set_shared_memory("output0_data", 1)

        outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT1"))
        outputs[-1].set_shared_memory("output1_data", 1)

        try:
            triton_client.infer(model_name="query", inputs=inputs, outputs=outputs)
            self.assertTrue(False, "expect error with query information")
        except InferenceServerException as ex:
            self.assertTrue("OUTPUT0 CPU 0" in ex.message())
            self.assertTrue("OUTPUT1 CPU 0" in ex.message())

        cudashm.destroy_shared_memory_region(shm_op0_handle)
        cudashm.destroy_shared_memory_region(shm_op1_handle)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()


if __name__ == "__main__":
    unittest.main()
