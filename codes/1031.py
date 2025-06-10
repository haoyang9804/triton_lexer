import sys

sys.path.append("../common")

import os
import queue
import threading
import time
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._response_queue = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._response_queue.put(error)
    else:
        user_data._response_queue.put(result)


class DecoupledTest(tu.TestResultCollector):
    def setUp(self):
        self.trials_ = [
            ("repeat_int32", None),
            ("simple_repeat", None),
            ("sequence_repeat", None),
            ("fan_repeat", self._fan_validate),
            ("repeat_square", self._nested_validate),
            ("nested_square", self._nested_validate),
        ]
        self.model_name_ = "repeat_int32"

        self.inputs_ = []
        self.inputs_.append(grpcclient.InferInput("IN", [1], "INT32"))
        self.inputs_.append(grpcclient.InferInput("DELAY", [1], "UINT32"))
        self.inputs_.append(grpcclient.InferInput("WAIT", [1], "UINT32"))

        self.outputs_ = []
        self.outputs_.append(grpcclient.InferRequestedOutput("OUT"))
        self.outputs_.append(grpcclient.InferRequestedOutput("IDX"))

        self.requested_outputs_ = self.outputs_

    def _stream_infer_with_params(
        self,
        request_count,
        request_delay,
        _,
        delay_data,
        delay_factor,
        user_data,
        result_dict,
    ):
        with grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        ) as triton_client:

            if "TRITONSERVER_GRPC_STATUS_FLAG" in os.environ:
                metadata = {"triton_grpc_error": "true"}
                triton_client.start_stream(
                    callback=partial(callback, user_data), headers=metadata
                )
            else:
                triton_client.start_stream(callback=partial(callback, user_data))

            for i in range(request_count):
                time.sleep((request_delay / 1000))
                self.inputs_[1].set_data_from_numpy(delay_data)
                triton_client.async_stream_infer(
                    model_name=self.model_name_,
                    inputs=self.inputs_,
                    request_id=str(i),
                    outputs=self.requested_outputs_,
                    enable_empty_final_response=True,
                )

                delay_data = delay_data * delay_factor
                delay_data = delay_data.astype(np.uint32)

            recv_count = 0
            completed_requests = 0
            while completed_requests < request_count:
                data_item = user_data._response_queue.get()
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    response = data_item.get_response()

                    if not response.id:
                        raise ValueError(
                            "No response id found. Was a request_id provided?"
                        )

                    if response.parameters.get("triton_final_response").bool_param:
                        completed_requests += 1

                    if response.outputs:
                        if response.id not in result_dict:
                            result_dict[response.id] = []
                        result_dict[response.id].append((recv_count, data_item))
                        recv_count += 1

    def _stream_infer(
        self,
        request_count,
        request_delay,
        expected_count,
        delay_data,
        delay_factor,
        user_data,
        result_dict,
    ):
        with grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        ) as triton_client:

            if "TRITONSERVER_GRPC_STATUS_FLAG" in os.environ:
                metadata = {"triton_grpc_error": "true"}
                triton_client.start_stream(
                    callback=partial(callback, user_data), headers=metadata
                )
            else:
                triton_client.start_stream(callback=partial(callback, user_data))

            for i in range(request_count):
                time.sleep((request_delay / 1000))
                self.inputs_[1].set_data_from_numpy(delay_data)
                triton_client.async_stream_infer(
                    model_name=self.model_name_,
                    inputs=self.inputs_,
                    request_id=str(i),
                    outputs=self.requested_outputs_,
                )

                delay_data = delay_data * delay_factor
                delay_data = delay_data.astype(np.uint32)

            recv_count = 0
            while recv_count < expected_count:
                data_item = user_data._response_queue.get()
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    this_id = data_item.get_response().id
                    if this_id not in result_dict:
                        result_dict[this_id] = []
                    result_dict[this_id].append((recv_count, data_item))

                recv_count += 1

    def _fan_validate(self, result_list, data_offset, repeat_count):

        self.assertEqual(len(result_list), repeat_count)
        expected_data = 2 * data_offset
        for j in range(len(result_list)):
            this_data = result_list[j][1].as_numpy("OUT")
            self.assertEqual(len(this_data), 1)
            self.assertEqual(this_data[0], expected_data)
            expected_data += 2

    def _nested_validate(self, result_list, data_offset, repeat_count):

        expected_len = sum(x for x in range(data_offset, data_offset + repeat_count))
        self.assertEqual(len(result_list), expected_len)
        expected_data = data_offset
        expected_count = expected_data
        for j in range(len(result_list)):
            this_data = result_list[j][1].as_numpy("OUT")
            self.assertEqual(len(this_data), 1)
            self.assertEqual(this_data[0], expected_data)
            expected_count -= 1
            if expected_count == 0:
                expected_data += 1
                expected_count = expected_data

    def _decoupled_infer(
        self,
        request_count,
        request_delay=0,
        repeat_count=1,
        data_offset=100,
        delay_time=1000,
        delay_factor=1,
        wait_time=500,
        order_sequence=None,
        validate_fn=None,
    ):

        input_data = np.arange(
            start=data_offset, stop=data_offset + repeat_count, dtype=np.int32
        )
        self.inputs_[0].set_shape([repeat_count])
        self.inputs_[0].set_data_from_numpy(input_data)

        delay_data = (np.ones([repeat_count], dtype=np.uint32)) * delay_time
        self.inputs_[1].set_shape([repeat_count])

        wait_data = np.array([wait_time], dtype=np.uint32)
        self.inputs_[2].set_data_from_numpy(wait_data)

        self.requested_outputs_ = (
            self.outputs_ if validate_fn is None else self.outputs_[0:1]
        )

        for infer_helper in [self._stream_infer, self._stream_infer_with_params]:
            user_data = UserData()
            result_dict = {}

            try:
                if "square" not in self.model_name_:
                    expected_count = repeat_count * request_count
                else:
                    expected_count = (
                        sum(x for x in range(data_offset, data_offset + repeat_count))
                        * request_count
                    )
                infer_helper(
                    request_count,
                    request_delay,
                    expected_count,
                    delay_data,
                    delay_factor,
                    user_data,
                    result_dict,
                )
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            for i in range(request_count):
                this_id = str(i)
                if repeat_count != 0 and this_id not in result_dict.keys():
                    self.assertTrue(
                        False, "response for request id {} not received".format(this_id)
                    )
                elif repeat_count == 0 and this_id in result_dict.keys():
                    self.assertTrue(
                        False,
                        "received unexpected response for request id {}".format(
                            this_id
                        ),
                    )
                if repeat_count != 0:
                    if validate_fn is None:
                        self.assertEqual(len(result_dict[this_id]), repeat_count)
                        expected_data = data_offset
                        result_list = result_dict[this_id]
                        for j in range(len(result_list)):
                            if order_sequence is not None:
                                self.assertEqual(
                                    result_list[j][0], order_sequence[i][j]
                                )
                            this_data = result_list[j][1].as_numpy("OUT")
                            self.assertEqual(len(this_data), 1)
                            self.assertEqual(this_data[0], expected_data)
                            this_idx = result_list[j][1].as_numpy("IDX")
                            self.assertEqual(len(this_idx), 1)
                            self.assertEqual(this_idx[0], j)
                            expected_data += 1
                    else:
                        validate_fn(result_dict[this_id], data_offset, repeat_count)

    def test_one_to_none(self):

        for trial in self.trials_:
            self.model_name_ = trial[0]

            self._decoupled_infer(request_count=1, repeat_count=0, validate_fn=trial[1])

            self._decoupled_infer(request_count=5, repeat_count=0, validate_fn=trial[1])

    def test_one_to_one(self):

        for trial in self.trials_:
            self.model_name_ = trial[0]

            self._decoupled_infer(request_count=1, wait_time=500, validate_fn=trial[1])

            self._decoupled_infer(request_count=1, wait_time=2000, validate_fn=trial[1])

            self._decoupled_infer(request_count=5, wait_time=500, validate_fn=trial[1])

            self._decoupled_infer(request_count=5, wait_time=2000, validate_fn=trial[1])

    def test_one_to_many(self):

        self.assertFalse("TRITONSERVER_DELAY_GRPC_RESPONSE" in os.environ)

        for trial in self.trials_:
            self.model_name_ = trial[0]

            self._decoupled_infer(
                request_count=1, repeat_count=5, wait_time=500, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=1, repeat_count=5, wait_time=2000, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=1, repeat_count=5, wait_time=10000, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=5, repeat_count=5, wait_time=500, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=5, repeat_count=5, wait_time=2000, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=5, repeat_count=5, wait_time=10000, validate_fn=trial[1]
            )

    def test_one_to_multi_many(self):

        self.assertTrue("TRITONSERVER_DELAY_GRPC_RESPONSE" in os.environ)

        for trial in self.trials_:
            self.model_name_ = trial[0]

            self._decoupled_infer(
                request_count=1, repeat_count=5, wait_time=500, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=1, repeat_count=5, wait_time=8000, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=1, repeat_count=5, wait_time=20000, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=5, repeat_count=5, wait_time=500, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=5, repeat_count=5, wait_time=3000, validate_fn=trial[1]
            )

            self._decoupled_infer(
                request_count=5, repeat_count=5, wait_time=10000, validate_fn=trial[1]
            )

    def test_response_order(self):

        self.assertFalse("TRITONSERVER_DELAY_GRPC_RESPONSE" in os.environ)

        for trial in self.trials_:
            self.model_name_ = trial[0]

            self._decoupled_infer(
                request_count=2,
                request_delay=500,
                repeat_count=4,
                order_sequence=[[0, 2, 4, 6], [1, 3, 5, 7]],
                validate_fn=trial[1],
            )

            self._decoupled_infer(
                request_count=2,
                request_delay=500,
                repeat_count=4,
                delay_time=2000,
                delay_factor=0.1,
                order_sequence=[[4, 5, 6, 7], [0, 1, 2, 3]],
                validate_fn=trial[1],
            )

            self._decoupled_infer(
                request_count=2,
                request_delay=2500,
                repeat_count=4,
                delay_time=2000,
                delay_factor=0.1,
                order_sequence=[[0, 5, 6, 7], [1, 2, 3, 4]],
                validate_fn=trial[1],
            )

            self._decoupled_infer(
                request_count=2,
                request_delay=100,
                repeat_count=4,
                delay_time=500,
                delay_factor=10,
                order_sequence=[[0, 1, 2, 3], [4, 5, 6, 7]],
                validate_fn=trial[1],
            )

            self._decoupled_infer(
                request_count=2,
                request_delay=750,
                repeat_count=4,
                delay_time=500,
                delay_factor=10,
                order_sequence=[[0, 1, 2, 3], [4, 5, 6, 7]],
                validate_fn=trial[1],
            )

    def _no_streaming_helper(self, protocol):
        data_offset = 100
        repeat_count = 1
        delay_time = 1000
        wait_time = 2000

        input_data = np.arange(
            start=data_offset, stop=data_offset + repeat_count, dtype=np.int32
        )
        delay_data = (np.ones([repeat_count], dtype=np.uint32)) * delay_time
        wait_data = np.array([wait_time], dtype=np.uint32)

        if protocol == "grpc":

            this_inputs = self.inputs_
            this_outputs = self.outputs_
        else:
            this_inputs = []
            this_inputs.append(httpclient.InferInput("IN", [repeat_count], "INT32"))
            this_inputs.append(httpclient.InferInput("DELAY", [1], "UINT32"))
            this_inputs.append(httpclient.InferInput("WAIT", [1], "UINT32"))
            this_outputs = []
            this_outputs.append(httpclient.InferRequestedOutput("OUT"))

        this_inputs[0].set_shape([repeat_count])
        this_inputs[0].set_data_from_numpy(input_data)

        this_inputs[1].set_shape([repeat_count])
        this_inputs[1].set_data_from_numpy(delay_data)

        this_inputs[2].set_data_from_numpy(wait_data)

        if protocol == "grpc":
            triton_client = grpcclient.InferenceServerClient(
                url="localhost:8001", verbose=True
            )
        else:
            triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=True
            )

        with self.assertRaises(InferenceServerException) as cm:
            triton_client.infer(
                model_name=self.model_name_, inputs=this_inputs, outputs=this_outputs
            )

        self.assertIn(
            "doesn't support models with decoupled transaction policy",
            str(cm.exception),
        )

    def test_no_streaming(self):

        self._no_streaming_helper("grpc")
        self._no_streaming_helper("http")

    def test_wrong_shape(self):

        data_offset = 100
        repeat_count = 1
        delay_time = 1000
        wait_time = 2000

        input_data = np.arange(
            start=data_offset, stop=data_offset + repeat_count, dtype=np.int32
        )
        delay_data = (np.ones([repeat_count + 1], dtype=np.uint32)) * delay_time
        wait_data = np.array([wait_time], dtype=np.uint32)

        self.inputs_[0].set_shape([repeat_count])
        self.inputs_[0].set_data_from_numpy(input_data)

        self.inputs_[1].set_shape([repeat_count + 1])
        self.inputs_[1].set_data_from_numpy(delay_data)

        self.inputs_[2].set_data_from_numpy(wait_data)

        user_data = UserData()
        result_dict = {}

        with self.assertRaises(InferenceServerException) as cm:
            self._stream_infer(
                1, 0, repeat_count, delay_data, 1, user_data, result_dict
            )

        self.assertIn(
            "expected IN and DELAY shape to match, got [1] and [2]", str(cm.exception)
        )


class NonDecoupledTest(tu.TestResultCollector):
    def setUp(self):
        self.model_name_ = "repeat_int32"
        self.data_matrix = [
            ([1], [0], [0]),
            ([1], [4000], [2000]),
            ([1], [2000], [4000]),
        ]

        self.callback_error = None
        self.callback_result = None
        self.callback_invoked_event = threading.Event()

    def _input_data(self, in_value, delay_value, wait_value):
        return {
            "IN": np.array(in_value, dtype=np.int32),
            "DELAY": np.array(delay_value, dtype=np.uint32),
            "WAIT": np.array(wait_value, dtype=np.uint32),
        }

    def _async_callback(self, result, error):

        self.callback_error = error
        self.callback_result = result
        self.callback_invoked_event.set()

    def test_grpc(self):
        for in_value, delay_value, wait_value in self.data_matrix:
            with self.subTest(IN=in_value, DELAY=delay_value, WAIT=wait_value):
                input_data = self._input_data(in_value, delay_value, wait_value)
                inputs = [
                    grpcclient.InferInput("IN", [1], "INT32").set_data_from_numpy(
                        input_data["IN"]
                    ),
                    grpcclient.InferInput("DELAY", [1], "UINT32").set_data_from_numpy(
                        input_data["DELAY"]
                    ),
                    grpcclient.InferInput("WAIT", [1], "UINT32").set_data_from_numpy(
                        input_data["WAIT"]
                    ),
                ]

                triton_client = grpcclient.InferenceServerClient(
                    url="localhost:8001", verbose=True
                )

                res = triton_client.infer(model_name=self.model_name_, inputs=inputs)
                self.assertEqual(1, res.as_numpy("OUT")[0])
                self.assertEqual(0, res.as_numpy("IDX")[0])

    def test_http(self):
        for in_value, delay_value, wait_value in self.data_matrix:
            with self.subTest(IN=in_value, DELAY=delay_value, WAIT=wait_value):
                input_data = self._input_data(in_value, delay_value, wait_value)
                inputs = [
                    httpclient.InferInput("IN", [1], "INT32").set_data_from_numpy(
                        input_data["IN"]
                    ),
                    httpclient.InferInput("DELAY", [1], "UINT32").set_data_from_numpy(
                        input_data["DELAY"]
                    ),
                    httpclient.InferInput("WAIT", [1], "UINT32").set_data_from_numpy(
                        input_data["WAIT"]
                    ),
                ]

                triton_client = httpclient.InferenceServerClient(
                    url="localhost:8000", verbose=True
                )

                res = triton_client.infer(model_name=self.model_name_, inputs=inputs)
                self.assertEqual(1, res.as_numpy("OUT")[0])
                self.assertEqual(0, res.as_numpy("IDX")[0])

    def test_grpc_async(self):
        for in_value, delay_value, wait_value in self.data_matrix:
            with self.subTest(IN=in_value, DELAY=delay_value, WAIT=wait_value):
                input_data = self._input_data(in_value, delay_value, wait_value)
                inputs = [
                    grpcclient.InferInput("IN", [1], "INT32").set_data_from_numpy(
                        input_data["IN"]
                    ),
                    grpcclient.InferInput("DELAY", [1], "UINT32").set_data_from_numpy(
                        input_data["DELAY"]
                    ),
                    grpcclient.InferInput("WAIT", [1], "UINT32").set_data_from_numpy(
                        input_data["WAIT"]
                    ),
                ]

                triton_client = grpcclient.InferenceServerClient(
                    url="localhost:8001",
                    verbose=True,
                )

                self.callback_error = None
                self.callback_result = None
                self.callback_invoked_event.clear()

                try:
                    triton_client.async_infer(
                        model_name=self.model_name_,
                        inputs=inputs,
                        callback=self._async_callback,
                    )
                except Exception as e:
                    self.fail(f"Failed to initiate async_infer: {e}")
                    continue

                self.assertTrue(
                    self.callback_invoked_event.wait(timeout=10),
                    "Callback not invoked within timeout.",
                )

                self.assertIsNone(
                    self.callback_error, f"Inference failed: {self.callback_error}"
                )
                self.assertIsNotNone(self.callback_result, "Inference result is None.")
                self.assertEqual(1, self.callback_result.as_numpy("OUT")[0])
                self.assertEqual(0, self.callback_result.as_numpy("IDX")[0])

                time.sleep(5)
                self.assertTrue(triton_client.is_model_ready(self.model_name_))

    def test_grpc_async_cancel(self):
        data_matrix = [
            ([1], [4000], [2000]),
            ([1], [2000], [4000]),
        ]

        for in_value, delay_value, wait_value in data_matrix:
            with self.subTest(IN=in_value, DELAY=delay_value, WAIT=wait_value):
                input_data = self._input_data(in_value, delay_value, wait_value)
                inputs = [
                    grpcclient.InferInput("IN", [1], "INT32").set_data_from_numpy(
                        input_data["IN"]
                    ),
                    grpcclient.InferInput("DELAY", [1], "UINT32").set_data_from_numpy(
                        input_data["DELAY"]
                    ),
                    grpcclient.InferInput("WAIT", [1], "UINT32").set_data_from_numpy(
                        input_data["WAIT"]
                    ),
                ]

                triton_client = grpcclient.InferenceServerClient(
                    url="localhost:8001",
                    verbose=True,
                )

                self.callback_error = None
                self.callback_result = None
                self.callback_invoked_event.clear()

                request_handle = None
                try:
                    request_handle = triton_client.async_infer(
                        model_name=self.model_name_,
                        inputs=inputs,
                        callback=self._async_callback,
                    )
                except Exception as e:
                    self.fail(f"Failed to initiate async_infer: {e}")
                    continue

                time.sleep(0.5)

                if request_handle:
                    try:
                        request_handle.cancel()
                    except Exception as e:
                        self.fail(f"Error calling request_handle.cancel(): {e}")
                        continue
                else:
                    self.fail("Invalid request_handle, cannot cancel.")
                    continue

                self.assertTrue(
                    self.callback_invoked_event.wait(timeout=10),
                    "Callback not invoked within timeout after cancellation.",
                )

                self.assertIsInstance(
                    self.callback_error,
                    InferenceServerException,
                    f"Unexpected error type: {type(self.callback_error)}",
                )
                self.assertIn(
                    "StatusCode.CANCELLED",
                    self.callback_error.status(),
                )

                time.sleep(5)
                self.assertTrue(triton_client.is_model_ready(self.model_name_))


if __name__ == "__main__":
    unittest.main()
