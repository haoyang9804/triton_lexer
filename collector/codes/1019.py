import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient.utils import triton_to_np_dtype


class BufferAttributesTest(tu.TestResultCollector):
    def test_buffer_attributes(self):
        model_name = "bls"

        clients = [
            httpclient.InferenceServerClient(url="localhost:8000"),
            grpcclient.InferenceServerClient(url="localhost:8001"),
        ]
        triton_clients = [httpclient, grpcclient]
        for i, client in enumerate(clients):

            client.unregister_system_shared_memory()
            client.unregister_cuda_shared_memory()

            triton_client = triton_clients[i]
            inputs = []
            outputs = []
            inputs.append(triton_client.InferInput("INPUT0", [1, 1000], "INT32"))

            input0_data = np.arange(start=0, stop=1000, dtype=np.int32)
            input0_data = np.expand_dims(input0_data, axis=0)

            input_byte_size = input0_data.size * input0_data.itemsize
            output_byte_size = input_byte_size

            shm_ip0_handle = cudashm.create_shared_memory_region(
                "input0_data", input_byte_size, 0
            )
            shm_op0_handle = cudashm.create_shared_memory_region(
                "output0_data", output_byte_size, 0
            )

            client.register_cuda_shared_memory(
                "input0_data",
                cudashm.get_raw_handle(shm_ip0_handle),
                0,
                input_byte_size,
            )
            client.register_cuda_shared_memory(
                "output0_data",
                cudashm.get_raw_handle(shm_op0_handle),
                0,
                input_byte_size,
            )

            cudashm.set_shared_memory_region(shm_ip0_handle, [input0_data])
            inputs[0].set_shared_memory("input0_data", input_byte_size)

            if triton_client is grpcclient:
                outputs.append(triton_client.InferRequestedOutput("OUTPUT0"))
                outputs[0].set_shared_memory("output0_data", output_byte_size)
            else:
                outputs.append(
                    triton_client.InferRequestedOutput("OUTPUT0", binary_data=True)
                )
                outputs[0].set_shared_memory("output0_data", output_byte_size)

            results = client.infer(
                model_name=model_name, inputs=inputs, outputs=outputs
            )

            output0 = results.get_output("OUTPUT0")
            self.assertIsNotNone(output0)
            if triton_client is grpcclient:
                output0_data = cudashm.get_contents_as_numpy(
                    shm_op0_handle, triton_to_np_dtype(output0.datatype), output0.shape
                )
            else:
                output0_data = cudashm.get_contents_as_numpy(
                    shm_op0_handle,
                    triton_to_np_dtype(output0["datatype"]),
                    output0["shape"],
                )
            self.assertTrue(np.all(output0_data == input0_data))


if __name__ == "__main__":
    unittest.main()
