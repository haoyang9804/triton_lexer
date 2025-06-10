import sys

sys.path.append("../common")

import unittest

import infer_util as iu
import numpy as np
import tritonclient.grpc as tritongrpcclient
import tritonclient.utils.shared_memory as shm
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


class InputValTest(unittest.TestCase):
    def test_input_validation_required_empty(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_all_required",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected 3 inputs but got 0 inputs for model 'input_all_required'. Got input(s) [], but missing required input(s) ['INPUT0','INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_optional_empty(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_optional",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected number of inputs between 3 and 4 but got 0 inputs for model 'input_optional'. Got input(s) [], but missing required input(s) ['INPUT0','INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_required_missing(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT0", [1], "FP32"))

        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_all_required",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected 3 inputs but got 1 inputs for model 'input_all_required'. Got input(s) ['INPUT0'], but missing required input(s) ['INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_optional(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput("INPUT0", [1], "FP32"))

        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))

        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name="input_optional",
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "expected number of inputs between 3 and 4 but got 1 inputs for model 'input_optional'. Got input(s) ['INPUT0'], but missing required input(s) ['INPUT1','INPUT2']. Please provide all required input(s).",
            err_str,
        )

    def test_input_validation_all_optional(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        result = triton_client.infer(
            model_name="input_all_optional",
            inputs=inputs,
        )
        response = result.get_response()
        self.assertIn(str(response.outputs[0].name), "OUTPUT0")


class InputShapeTest(unittest.TestCase):
    def test_input_shape_validation(self):
        input_size = 8
        model_name = "pt_identity"
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

        input_data = np.arange(input_size)[None].astype(np.float32)
        inputs = [
            tritongrpcclient.InferInput(
                "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)
        triton_client.infer(model_name=model_name, inputs=inputs)

        input_data = np.arange(input_size + 2)[None].astype(np.float32)
        inputs = [
            tritongrpcclient.InferInput(
                "INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input_data)

        inputs[0].set_shape((1, input_size))
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(
                model_name=model_name,
                inputs=inputs,
            )
        err_str = str(e.exception)
        self.assertIn(
            "input byte size mismatch for input 'INPUT0' for model 'pt_identity'. Expected 32, got 40",
            err_str,
        )

    def test_input_string_shape_validation(self):
        input_size = 16
        model_name = "onnx_object_int32_int32"
        np_dtype_string = np.dtype(object)
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")

        def get_input_array(input_size, np_dtype):
            rinput_dtype = iu._range_repr_dtype(np_dtype)
            input_array = np.random.randint(
                low=0, high=127, size=(1, input_size), dtype=rinput_dtype
            )

            inn = np.array(
                [str(x) for x in input_array.reshape(input_array.size)], dtype=object
            )
            input_array = inn.reshape(input_array.shape)

            inputs = []
            inputs.append(
                tritongrpcclient.InferInput(
                    "INPUT0", input_array.shape, np_to_triton_dtype(np_dtype)
                )
            )
            inputs.append(
                tritongrpcclient.InferInput(
                    "INPUT1", input_array.shape, np_to_triton_dtype(np_dtype)
                )
            )

            inputs[0].set_data_from_numpy(input_array)
            inputs[1].set_data_from_numpy(input_array)
            return inputs

        inputs = get_input_array(input_size - 2, np_dtype_string)

        inputs[0].set_shape((1, input_size))
        inputs[1].set_shape((1, input_size))
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(model_name=model_name, inputs=inputs)
        err_str = str(e.exception)
        self.assertIn(
            f"expected {input_size} string elements for inference input 'INPUT1' for model '{model_name}', got {input_size-2}",
            err_str,
        )

        inputs = get_input_array(input_size + 2, np_dtype_string)

        inputs[0].set_shape((1, input_size))
        inputs[1].set_shape((1, input_size))
        with self.assertRaises(InferenceServerException) as e:
            triton_client.infer(model_name=model_name, inputs=inputs)
        err_str = str(e.exception)
        self.assertIn(
            f"unexpected number of string elements {input_size+1} for inference input 'INPUT1' for model '{model_name}', expecting {input_size}",
            err_str,
        )

    def test_wrong_input_shape_tensor_size(self):
        def inference_helper(model_name, batch_size=1):
            triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
            if batch_size > 1:
                dummy_input_data = np.random.rand(batch_size, 32, 32).astype(np.float32)
            else:
                dummy_input_data = np.random.rand(32, 32).astype(np.float32)
            shape_tensor_data = np.asarray([4, 4], dtype=np.int32)

            input_byte_size = (shape_tensor_data.size - 1) * np.dtype(np.int32).itemsize

            input_shm_handle = shm.create_shared_memory_region(
                "INPUT0_SHM",
                "/INPUT0_SHM",
                input_byte_size,
            )

            shm.set_shared_memory_region(
                input_shm_handle,
                [
                    shape_tensor_data[: input_byte_size // np.dtype(np.int32).itemsize],
                ],
            )
            triton_client.register_system_shared_memory(
                "INPUT0_SHM",
                "/INPUT0_SHM",
                input_byte_size,
            )

            inputs = [
                tritongrpcclient.InferInput(
                    "DUMMY_INPUT0",
                    dummy_input_data.shape,
                    np_to_triton_dtype(np.float32),
                ),
                tritongrpcclient.InferInput(
                    "INPUT0",
                    shape_tensor_data.shape,
                    np_to_triton_dtype(np.int32),
                ),
            ]
            inputs[0].set_data_from_numpy(dummy_input_data)
            inputs[1].set_shared_memory("INPUT0_SHM", input_byte_size)

            outputs = [
                tritongrpcclient.InferRequestedOutput("DUMMY_OUTPUT0"),
                tritongrpcclient.InferRequestedOutput("OUTPUT0"),
            ]

            try:

                with self.assertRaises(InferenceServerException) as e:
                    triton_client.infer(
                        model_name=model_name, inputs=inputs, outputs=outputs
                    )
                err_str = str(e.exception)
                correct_input_byte_size = (
                    shape_tensor_data.size * np.dtype(np.int32).itemsize
                )
                self.assertIn(
                    f"input byte size mismatch for input 'INPUT0' for model '{model_name}'. Expected {correct_input_byte_size}, got {input_byte_size}",
                    err_str,
                )
            finally:
                shm.destroy_shared_memory_region(input_shm_handle)
                triton_client.unregister_system_shared_memory("INPUT0_SHM")

        inference_helper(model_name="plan_nobatch_zero_1_float32_int32")
        inference_helper(model_name="plan_zero_1_float32_int32", batch_size=8)


if __name__ == "__main__":
    unittest.main()
