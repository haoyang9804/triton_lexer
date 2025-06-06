import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as tritongrpcclient
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException


class ClientNoBatchTest(tu.TestResultCollector):
    def test_nobatch_request_for_batching_model(self):
        input_size = 16

        tensor_shape = (input_size,)
        for protocol in ["http", "grpc"]:
            model_name = tu.get_model_name("onnx", np.int32, np.int8, np.int8)
            in0 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)
            in1 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)

            inputs = []
            outputs = []
            if protocol == "http":
                triton_client = tritonhttpclient.InferenceServerClient(
                    url="localhost:8000", verbose=True
                )
                inputs.append(
                    tritonhttpclient.InferInput("INPUT0", tensor_shape, "INT32")
                )
                inputs.append(
                    tritonhttpclient.InferInput("INPUT1", tensor_shape, "INT32")
                )
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT0"))
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT1"))
            else:
                triton_client = tritongrpcclient.InferenceServerClient(
                    url="localhost:8001", verbose=True
                )
                inputs.append(
                    tritongrpcclient.InferInput("INPUT0", tensor_shape, "INT32")
                )
                inputs.append(
                    tritongrpcclient.InferInput("INPUT1", tensor_shape, "INT32")
                )
                outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT0"))
                outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT1"))

            inputs[0].set_data_from_numpy(in0)
            inputs[1].set_data_from_numpy(in1)

            try:
                _ = triton_client.infer(model_name, inputs, outputs=outputs)
                self.assertTrue(
                    False, "expected failure with no batch request for batching model"
                )
            except InferenceServerException as ex:
                pass

    def test_batch_request_for_nobatching_model(self):
        input_size = 16

        tensor_shape = (1, input_size)
        for protocol in ["http", "grpc"]:
            model_name = tu.get_model_name("onnx_nobatch", np.int32, np.int8, np.int8)
            in0 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)
            in1 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)

            inputs = []
            outputs = []
            if protocol == "http":
                triton_client = tritonhttpclient.InferenceServerClient(
                    url="localhost:8000", verbose=True
                )
                inputs.append(
                    tritonhttpclient.InferInput("INPUT0", tensor_shape, "INT32")
                )
                inputs.append(
                    tritonhttpclient.InferInput("INPUT1", tensor_shape, "INT32")
                )
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT0"))
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT1"))
            else:
                triton_client = tritongrpcclient.InferenceServerClient(
                    url="localhost:8001", verbose=True
                )
                inputs.append(
                    tritongrpcclient.InferInput("INPUT0", tensor_shape, "INT32")
                )
                inputs.append(
                    tritongrpcclient.InferInput("INPUT1", tensor_shape, "INT32")
                )
                outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT0"))
                outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT1"))

            inputs[0].set_data_from_numpy(in0)
            inputs[1].set_data_from_numpy(in1)

            try:
                _ = triton_client.infer(model_name, inputs, outputs=outputs)
                self.assertTrue(
                    False,
                    "expected failure with batched request for non-batching model",
                )
            except InferenceServerException as ex:
                pass

    def test_nobatch_request_for_nonbatching_model(self):
        input_size = 16

        tensor_shape = (input_size,)
        for protocol in ["http", "grpc"]:
            model_name = tu.get_model_name("onnx_nobatch", np.int32, np.int8, np.int8)
            in0 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)
            in1 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)

            inputs = []
            outputs = []
            if protocol == "http":
                triton_client = tritonhttpclient.InferenceServerClient(
                    url="localhost:8000", verbose=True
                )
                inputs.append(
                    tritonhttpclient.InferInput("INPUT0", tensor_shape, "INT32")
                )
                inputs.append(
                    tritonhttpclient.InferInput("INPUT1", tensor_shape, "INT32")
                )
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT0"))
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT1"))
            else:
                triton_client = tritongrpcclient.InferenceServerClient(
                    url="localhost:8001", verbose=True
                )
                inputs.append(
                    tritongrpcclient.InferInput("INPUT0", tensor_shape, "INT32")
                )
                inputs.append(
                    tritongrpcclient.InferInput("INPUT1", tensor_shape, "INT32")
                )
                outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT0"))
                outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT1"))

            inputs[0].set_data_from_numpy(in0)
            inputs[1].set_data_from_numpy(in1)

            results = triton_client.infer(model_name, inputs, outputs=outputs)

    def test_batch_request_for_batching_model(self):
        input_size = 16

        tensor_shape = (1, input_size)
        for protocol in ["http", "grpc"]:
            model_name = tu.get_model_name("onnx", np.int32, np.int8, np.int8)
            in0 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)
            in1 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)

            inputs = []
            outputs = []
            if protocol == "http":
                triton_client = tritonhttpclient.InferenceServerClient(
                    url="localhost:8000", verbose=True
                )
                inputs.append(
                    tritonhttpclient.InferInput("INPUT0", tensor_shape, "INT32")
                )
                inputs.append(
                    tritonhttpclient.InferInput("INPUT1", tensor_shape, "INT32")
                )
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT0"))
                outputs.append(tritonhttpclient.InferRequestedOutput("OUTPUT1"))
            else:
                triton_client = tritongrpcclient.InferenceServerClient(
                    url="localhost:8001", verbose=True
                )
                inputs.append(
                    tritongrpcclient.InferInput("INPUT0", tensor_shape, "INT32")
                )
                inputs.append(
                    tritongrpcclient.InferInput("INPUT1", tensor_shape, "INT32")
                )
                outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT0"))
                outputs.append(tritongrpcclient.InferRequestedOutput("OUTPUT1"))

            inputs[0].set_data_from_numpy(in0)
            inputs[1].set_data_from_numpy(in1)

            results = triton_client.infer(model_name, inputs, outputs=outputs)


if __name__ == "__main__":
    unittest.main()
