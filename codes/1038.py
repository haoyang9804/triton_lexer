import sys

sys.path.append("../common")

import os
import queue
import signal
import time
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._response_queue = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._response_queue.put(error)
    else:
        user_data._response_queue.put(result)


class CleanUpTest(tu.TestResultCollector):
    SERVER_PID = None

    def setUp(self):
        self.decoupled_model_name_ = "repeat_int32"
        self.identity_model_name_ = "custom_zero_1_float32"
        self.repeat_non_decoupled_model_name = "repeat_int32_non_decoupled"

    def _prepare_inputs_and_outputs(self, kind):
        if kind in ("decoupled_streaming", "non_decoupled_streaming"):
            self.inputs_ = []
            self.inputs_.append(grpcclient.InferInput("IN", [1], "INT32"))
            self.inputs_.append(grpcclient.InferInput("DELAY", [1], "UINT32"))
            self.inputs_.append(grpcclient.InferInput("WAIT", [1], "UINT32"))

            self.outputs_ = []
            self.outputs_.append(grpcclient.InferRequestedOutput("OUT"))
            self.outputs_.append(grpcclient.InferRequestedOutput("IDX"))
            self.requested_outputs_ = self.outputs_
        elif kind in ("simple", "streaming"):
            self.inputs_ = []
            self.inputs_.append(grpcclient.InferInput("INPUT0", [1, 1], "FP32"))

            self.outputs_ = []
            self.outputs_.append(grpcclient.InferRequestedOutput("OUTPUT0"))
            self.requested_outputs_ = self.outputs_
        else:
            raise ValueError("Unsupported kind specified to prepare inputs/outputs")

    def _simple_infer(
        self,
        request_count,
        cancel_response_idx=None,
        client_timeout_pair=None,
        kill_server=None,
    ):
        with grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        ) as triton_client:
            self._prepare_inputs_and_outputs("simple")

            input_data = np.array([[1.0]], dtype=np.float32)
            self.inputs_[0].set_data_from_numpy(input_data)

            user_data = UserData()

            futures = []
            timeout_idx = None
            timeout_value = None
            if client_timeout_pair:
                timeout_idx, timeout_value = client_timeout_pair
            for i in range(request_count):
                if kill_server == i:
                    os.kill(int(self.SERVER_PID), signal.SIGINT)
                this_timeout = None
                if timeout_idx == i:
                    this_timeout = timeout_value
                futures.append(
                    triton_client.async_infer(
                        model_name=self.identity_model_name_,
                        inputs=self.inputs_,
                        request_id=str(i),
                        callback=partial(callback, user_data),
                        outputs=self.requested_outputs_,
                        client_timeout=this_timeout,
                    )
                )

            if cancel_response_idx is not None:
                futures[cancel_response_idx].cancel()

            responses = []
            while len(responses) < len(futures):
                data_item = user_data._response_queue.get()
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    responses.append(data_item)

            for response in responses:
                output0_data = response.as_numpy("OUTPUT0")
                self.assertTrue(np.array_equal(input_data, output0_data))

    def _stream_infer_with_params(
        self,
        request_count,
        request_delay,
        _,
        user_data,
        result_dict,
        delay_data=None,
        delay_factor=None,
        cancel_response_idx=None,
        stream_timeout=None,
        kill_server=None,
    ):
        with grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        ) as triton_client:

            if "TRITONSERVER_GRPC_STATUS_FLAG" in os.environ:
                metadata = {"triton_grpc_error": "true"}
                triton_client.start_stream(
                    callback=partial(callback, user_data),
                    stream_timeout=stream_timeout,
                    headers=metadata,
                )
            else:
                triton_client.start_stream(
                    callback=partial(callback, user_data), stream_timeout=stream_timeout
                )

            for i in range(request_count):
                time.sleep((request_delay / 1000))
                self.inputs_[1].set_data_from_numpy(delay_data)
                if kill_server == i:
                    os.kill(int(self.SERVER_PID), signal.SIGINT)
                triton_client.async_stream_infer(
                    model_name=self.decoupled_model_name_,
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
                if cancel_response_idx == recv_count:
                    triton_client.stop_stream(cancel_requests=True)
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
        user_data,
        result_dict,
        delay_data=None,
        delay_factor=None,
        cancel_response_idx=None,
        stream_timeout=None,
        kill_server=None,
    ):
        with grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        ) as triton_client:

            if "TRITONSERVER_GRPC_STATUS_FLAG" in os.environ:
                metadata = {"triton_grpc_error": "true"}
                triton_client.start_stream(
                    callback=partial(callback, user_data),
                    stream_timeout=stream_timeout,
                    headers=metadata,
                )
            else:
                triton_client.start_stream(
                    callback=partial(callback, user_data), stream_timeout=stream_timeout
                )

            for i in range(request_count):
                time.sleep((request_delay / 1000))
                model_name = self.identity_model_name_
                if delay_data is not None:
                    model_name = self.decoupled_model_name_
                    self.inputs_[1].set_data_from_numpy(delay_data)
                if kill_server == i:
                    os.kill(int(self.SERVER_PID), signal.SIGINT)
                triton_client.async_stream_infer(
                    model_name=model_name,
                    inputs=self.inputs_,
                    request_id=str(i),
                    outputs=self.requested_outputs_,
                )
                if (delay_data is not None) and (delay_factor is not None):

                    delay_data = delay_data * delay_factor
                    delay_data = delay_data.astype(np.uint32)

            recv_count = 0
            while recv_count < expected_count:
                if cancel_response_idx == recv_count:
                    triton_client.stop_stream(cancel_requests=True)
                data_item = user_data._response_queue.get()
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    this_id = data_item.get_response().id
                    if this_id not in result_dict:
                        result_dict[this_id] = []
                    result_dict[this_id].append((recv_count, data_item))

                recv_count += 1

    def _streaming_infer(
        self,
        request_count,
        request_delay=0,
        cancel_response_idx=None,
        stream_timeout=None,
        kill_server=None,
        should_error=True,
    ):
        self._prepare_inputs_and_outputs("streaming")

        input_data = np.array([[1.0]], dtype=np.float32)
        self.inputs_[0].set_data_from_numpy(input_data)

        user_data = UserData()
        result_dict = {}

        try:
            expected_count = request_count
            self._stream_infer(
                request_count,
                request_delay,
                expected_count,
                user_data,
                result_dict,
                cancel_response_idx=cancel_response_idx,
                stream_timeout=stream_timeout,
                kill_server=kill_server,
            )
        except Exception as ex:
            if cancel_response_idx or stream_timeout or should_error:
                raise ex
            self.assertTrue(False, "unexpected error {}".format(ex))

        for i in range(request_count):
            this_id = str(i)
            if this_id not in result_dict.keys():
                self.assertTrue(
                    False, "response for request id {} not received".format(this_id)
                )
            self.assertEqual(len(result_dict[this_id]), 1)
            result = result_dict[this_id][0][1]
            output0_data = result.as_numpy("OUTPUT0")
            self.assertTrue(np.array_equal(input_data, output0_data))

    def _decoupled_infer(
        self,
        request_count,
        request_delay=0,
        repeat_count=1,
        data_offset=100,
        delay_time=1000,
        delay_factor=1,
        wait_time=500,
        cancel_response_idx=None,
        stream_timeout=None,
        kill_server=None,
        should_error=True,
        infer_helper_map=[True, True],
    ):
        self._prepare_inputs_and_outputs(kind="decoupled_streaming")

        input_data = np.arange(
            start=data_offset, stop=data_offset + repeat_count, dtype=np.int32
        )
        self.inputs_[0].set_shape([repeat_count])
        self.inputs_[0].set_data_from_numpy(input_data)

        delay_data = (np.ones([repeat_count], dtype=np.uint32)) * delay_time
        self.inputs_[1].set_shape([repeat_count])

        wait_data = np.array([wait_time], dtype=np.uint32)
        self.inputs_[2].set_data_from_numpy(wait_data)

        infer_helpers = []
        if infer_helper_map[0]:
            infer_helpers.append(self._stream_infer)
        if infer_helper_map[1]:
            infer_helpers.append(self._stream_infer_with_params)

        for infer_helper in infer_helpers:
            user_data = UserData()
            result_dict = {}

            try:
                expected_count = repeat_count * request_count
                infer_helper(
                    request_count,
                    request_delay,
                    expected_count,
                    user_data,
                    result_dict,
                    delay_data,
                    delay_factor,
                    cancel_response_idx,
                    stream_timeout,
                    kill_server,
                )
            except Exception as ex:
                if cancel_response_idx or stream_timeout or should_error:
                    raise ex
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
                    self.assertEqual(len(result_dict[this_id]), repeat_count)
                    expected_data = data_offset
                    result_list = result_dict[this_id]
                    for j in range(len(result_list)):
                        this_data = result_list[j][1].as_numpy("OUT")
                        self.assertEqual(len(this_data), 1)
                        self.assertEqual(this_data[0], expected_data)
                        this_idx = result_list[j][1].as_numpy("IDX")
                        self.assertEqual(len(this_idx), 1)
                        self.assertEqual(this_idx[0], j)
                        expected_data += 1

    def test_simple_infer(self):

        self._simple_infer(request_count=10)

    def test_simple_infer_cancellation(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._simple_infer(request_count=10, cancel_response_idx=5)
        self.assertIn("Locally cancelled by application!", str(cm.exception))

    def test_simple_infer_timeout(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._simple_infer(request_count=10, client_timeout_pair=[5, 0.1])
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_simple_infer_error_status(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._simple_infer(request_count=10)
        self.assertIn(
            "This protocol is restricted, expecting header 'triton-grpc-protocol-infer-key'",
            str(cm.exception),
        )

    def test_simple_infer_shutdownserver(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._simple_infer(request_count=20, kill_server=5)

    def test_streaming_infer(self):

        self._streaming_infer(request_count=10)

    def test_streaming_cancellation(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._streaming_infer(request_count=10, cancel_response_idx=5)
        self.assertIn("Locally cancelled by application!", str(cm.exception))

    def test_streaming_timeout(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._streaming_infer(request_count=10, request_delay=1, stream_timeout=2)
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_streaming_error_status(self):

        expected_exceptions = [
            "This protocol is restricted, expecting header 'triton-grpc-protocol-infer-key'",
            "The stream is no longer in valid state, the error detail is reported through provided callback. A new stream should be started after stopping the current stream.",
        ]
        with self.assertRaises(InferenceServerException) as cm:
            self._streaming_infer(request_count=10, should_error=True)

        exception_match = False
        for expected_exception in expected_exceptions:
            exception_match |= expected_exception in str(cm.exception)
        self.assertTrue(
            exception_match, "Raised unexpected exception {}".format(str(cm.exception))
        )

    def test_streaming_infer_shutdownserver(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._streaming_infer(
                request_count=10,
                request_delay=1,
                kill_server=5,
                should_error=True,
            )

    def test_decoupled_infer(self):

        self._decoupled_infer(request_count=10, repeat_count=10)

    def test_decoupled_cancellation(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(
                request_count=10, repeat_count=10, cancel_response_idx=5
            )
        self.assertIn("Locally cancelled by application!", str(cm.exception))

    def test_decoupled_timeout(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(
                request_count=10, repeat_count=10, request_delay=1, stream_timeout=2
            )
        self.assertIn("Deadline Exceeded", str(cm.exception))

    def test_decoupled_error_status(self):

        expected_exceptions = [
            "This protocol is restricted, expecting header 'triton-grpc-protocol-infer-key'",
            "The stream is no longer in valid state, the error detail is reported through provided callback. A new stream should be started after stopping the current stream.",
        ]
        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(request_count=10, repeat_count=10, should_error=True)

        exception_match = False
        for expected_exception in expected_exceptions:
            exception_match |= expected_exception in str(cm.exception)
        self.assertTrue(
            exception_match, "Raised unexpected exception {}".format(str(cm.exception))
        )

    def test_decoupled_infer_shutdownserver(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(
                request_count=10,
                repeat_count=10,
                request_delay=1,
                kill_server=5,
                should_error=True,
                infer_helper_map=[True, False],
            )

    def test_decoupled_infer_with_params_shutdownserver(self):

        with self.assertRaises(InferenceServerException) as cm:
            self._decoupled_infer(
                request_count=10,
                repeat_count=10,
                request_delay=1,
                kill_server=5,
                should_error=True,
                infer_helper_map=[False, True],
            )

    def test_decoupled_infer_complete(self):

        self._decoupled_infer(request_count=1, repeat_count=1, stream_timeout=16)

        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("Should not print this", server_log)

    def test_non_decoupled_streaming_multi_response(self):

        response_count = 4
        expected_response_count = 1
        expected_response_index = 0

        self._prepare_inputs_and_outputs("non_decoupled_streaming")

        data_offset = 100
        input_data = np.arange(
            start=data_offset, stop=data_offset + response_count, dtype=np.int32
        )
        self.inputs_[0].set_shape([response_count])
        self.inputs_[0].set_data_from_numpy(input_data)

        delay_data = np.zeros([response_count], dtype=np.uint32)
        self.inputs_[1].set_shape([response_count])
        self.inputs_[1].set_data_from_numpy(delay_data)

        wait_data = np.array([0], dtype=np.uint32)
        self.inputs_[2].set_data_from_numpy(wait_data)

        user_data = UserData()
        with grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        ) as client:

            if "TRITONSERVER_GRPC_STATUS_FLAG" in os.environ:
                metadata = {"triton_grpc_error": "true"}
                client.start_stream(
                    callback=partial(callback, user_data),
                    stream_timeout=16,
                    headers=metadata,
                )
            else:
                client.start_stream(
                    callback=partial(callback, user_data), stream_timeout=16
                )

            client.async_stream_infer(
                model_name=self.repeat_non_decoupled_model_name,
                inputs=self.inputs_,
                request_id="0",
                outputs=self.requested_outputs_,
            )

            client.stop_stream()

        actual_response_count = 0
        while not user_data._response_queue.empty():
            actual_response_count += 1
            data_item = user_data._response_queue.get()
            if type(data_item) == InferenceServerException:
                raise data_item
            else:
                response_idx = data_item.as_numpy("IDX")[0]
                self.assertEqual(response_idx, expected_response_index)
        self.assertEqual(actual_response_count, expected_response_count)


if __name__ == "__main__":
    CleanUpTest.SERVER_PID = os.environ.get("SERVER_PID", CleanUpTest.SERVER_PID)
    unittest.main()
