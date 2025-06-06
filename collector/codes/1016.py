import sys

sys.path.append("../common")

import queue
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class BatchInputTest(tu.TestResultCollector):
    def setUp(self):
        self.client = grpcclient.InferenceServerClient(url="localhost:8001")

        def callback(user_data, result, error):
            if error:
                user_data.put(error)
            else:
                user_data.put(result)

        self.client_callback = callback

    def set_inputs(self, shapes, input_name):
        self.dtype_ = np.float32
        self.inputs = []
        for shape in shapes:
            self.inputs.append(
                [grpcclient.InferInput(input_name, [1, shape[0]], "FP32")]
            )
            self.inputs[-1][0].set_data_from_numpy(
                np.full([1, shape[0]], shape[0], np.float32)
            )

    def set_inputs_for_batch_item(self, shapes, input_name):
        self.dtype_ = np.float32
        self.inputs = []
        for shape in shapes:
            self.inputs.append([grpcclient.InferInput(input_name, shape, "FP32")])
            self.inputs[-1][0].set_data_from_numpy(np.full(shape, shape[0], np.float32))

    def test_ragged_output(self):
        model_name = "ragged_io"

        self.set_inputs([[2], [4], [1], [3]], "INPUT0")
        user_data = queue.Queue()
        self.client.start_stream(callback=partial(self.client_callback, user_data))

        output_name = "OUTPUT0"
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for input in self.inputs:

                async_requests.append(
                    self.client.async_stream_infer(
                        model_name=model_name, inputs=input, outputs=outputs
                    )
                )

            expected_value_list = [[v] * v for v in [2, 4, 1, 3]]
            expected_value_list = [
                np.asarray([expected_value], dtype=np.float32)
                for expected_value in expected_value_list
            ]
            for idx in range(len(async_requests)):

                result = user_data.get()

                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value_list[idx]),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value_list[idx], output_data
                    ),
                )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self.client.stop_stream()

    def test_ragged_input(self):
        model_name = "ragged_acc_shape"
        self.set_inputs([[2], [4], [1], [3]], "RAGGED_INPUT")
        user_data = queue.Queue()
        self.client.start_stream(callback=partial(self.client_callback, user_data))

        output_name = "RAGGED_OUTPUT"
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        async_requests = []
        try:
            for input in self.inputs:

                async_requests.append(
                    self.client.async_stream_infer(
                        model_name=model_name, inputs=input, outputs=outputs
                    )
                )

            value_lists = [[v] * v for v in [2, 4, 1, 3]]
            expected_value = []
            for value_list in value_lists:
                expected_value += value_list
            expected_value = np.asarray([expected_value], dtype=np.float32)
            for idx in range(len(async_requests)):

                result = user_data.get()

                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data
                    ),
                )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self.client.stop_stream()

    def test_element_count(self):
        model_name = "ragged_element_count_acc_zero"
        self.set_inputs([[2], [4], [1], [3]], "RAGGED_INPUT")
        user_data = queue.Queue()
        self.client.start_stream(callback=partial(self.client_callback, user_data))

        output_name = "BATCH_AND_SIZE_OUTPUT"
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for input in self.inputs:

                async_requests.append(
                    self.client.async_stream_infer(
                        model_name=model_name, inputs=input, outputs=outputs
                    )
                )

            expected_value = np.asarray([[2, 4, 1, 3]], np.float32)
            for idx in range(len(async_requests)):

                result = user_data.get()

                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data
                    ),
                )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self.client.stop_stream()

    def test_accumulated_element_count(self):
        model_name = "ragged_acc_shape"
        self.set_inputs([[2], [4], [1], [3]], "RAGGED_INPUT")
        user_data = queue.Queue()
        self.client.start_stream(callback=partial(self.client_callback, user_data))

        output_name = "BATCH_AND_SIZE_OUTPUT"
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for input in self.inputs:

                async_requests.append(
                    self.client.async_stream_infer(
                        model_name=model_name, inputs=input, outputs=outputs
                    )
                )

            expected_value = np.asarray([[2, 6, 7, 10]], np.float32)
            for idx in range(len(async_requests)):

                result = user_data.get()

                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data
                    ),
                )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self.client.stop_stream()

    def test_accumulated_element_count_with_zero(self):
        model_name = "ragged_element_count_acc_zero"
        self.set_inputs([[2], [4], [1], [3]], "RAGGED_INPUT")
        user_data = queue.Queue()
        self.client.start_stream(callback=partial(self.client_callback, user_data))

        output_name = "BATCH_OUTPUT"
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for input in self.inputs:

                async_requests.append(
                    self.client.async_stream_infer(
                        model_name=model_name, inputs=input, outputs=outputs
                    )
                )

            expected_value = np.asarray([[0, 2, 6, 7, 10]], np.float32)
            for idx in range(len(async_requests)):

                result = user_data.get()

                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data
                    ),
                )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self.client.stop_stream()

    def test_max_element_count_as_shape(self):
        model_name = "ragged_acc_shape"
        self.set_inputs([[2], [4], [1], [3]], "RAGGED_INPUT")
        user_data = queue.Queue()
        self.client.start_stream(callback=partial(self.client_callback, user_data))

        output_name = "BATCH_OUTPUT"
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for input in self.inputs:

                async_requests.append(
                    self.client.async_stream_infer(
                        model_name=model_name, inputs=input, outputs=outputs
                    )
                )

            for idx in range(len(async_requests)):

                result = user_data.get()

                output_data = result.as_numpy(output_name)
                self.assertEqual(
                    output_data.shape,
                    (1, 4),
                    "Expect response {} to have shape to represent max element count {} among the batch , got {}".format(
                        idx, 4, output_data.shape
                    ),
                )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self.client.stop_stream()

    def test_batch_item_shape_flatten(self):

        self.set_inputs_for_batch_item(
            [[1, 4, 1], [1, 1, 2], [1, 1, 2], [1, 2, 2]], "RAGGED_INPUT"
        )

        model_name = "batch_item_flatten"
        user_data = queue.Queue()
        self.client.start_stream(callback=partial(self.client_callback, user_data))

        output_name = "BATCH_OUTPUT"
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for input in self.inputs:

                async_requests.append(
                    self.client.async_stream_infer(
                        model_name=model_name, inputs=input, outputs=outputs
                    )
                )

            expected_value = np.asarray([[4, 1, 1, 2, 1, 2, 2, 2]], np.float32)
            for idx in range(len(async_requests)):

                result = user_data.get()

                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data
                    ),
                )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self.client.stop_stream()

    def test_batch_item_shape(self):

        self.set_inputs_for_batch_item(
            [[2, 1, 2], [1, 1, 2], [1, 2, 2]], "RAGGED_INPUT"
        )

        expected_outputs = [
            np.array([[1.0, 2.0], [1.0, 2.0]]),
            np.array([[1.0, 2.0]]),
            np.array([[2.0, 2.0]]),
        ]

        model_name = "batch_item"
        user_data = queue.Queue()
        self.client.start_stream(callback=partial(self.client_callback, user_data))

        output_name = "BATCH_OUTPUT"
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for input in self.inputs:

                async_requests.append(
                    self.client.async_stream_infer(
                        model_name=model_name, inputs=input, outputs=outputs
                    )
                )

            for idx in range(len(async_requests)):

                result = user_data.get()

                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.allclose(output_data, expected_outputs[idx]),
                    "Expect response to have value:\n{}, got:\n{}\nEqual matrix:\n{}".format(
                        expected_outputs[idx],
                        output_data,
                        np.isclose(expected_outputs[idx], output_data),
                    ),
                )
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self.client.stop_stream()


if __name__ == "__main__":
    unittest.main()
