import queue
import time
import unittest


from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class RestrictedProtocolTest(unittest.TestCase):
    def setUp(self):
        self.client_ = grpcclient.InferenceServerClient(url="localhost:8001")
        self.model_name_ = "simple"
        self.prefix_ = "triton-grpc-protocol-"

    def test_sanity(self):
        self.client_.get_inference_statistics("simple")
        self.client_.get_inference_statistics(
            "simple", headers={self.prefix_ + "infer-key": "infer-value"}
        )

    def test_model_repository(self):
        with self.assertRaisesRegex(
            InferenceServerException, "This protocol is restricted"
        ):
            self.client_.unload_model(
                self.model_name_, headers={self.prefix_ + "infer-key": "infer-value"}
            )

        with self.assertRaisesRegex(
            InferenceServerException, "explicit model load / unload is not allowed"
        ):
            self.client_.unload_model(
                self.model_name_, headers={self.prefix_ + "admin-key": "admin-value"}
            )

    def test_health(self):
        with self.assertRaisesRegex(
            InferenceServerException, "This protocol is restricted"
        ):
            self.client_.is_server_live()
        self.client_.is_server_live({self.prefix_ + "infer-key": "infer-value"})

    def test_infer(self):

        inputs = [
            grpcclient.InferInput("INPUT0", [1, 16], "INT32"),
            grpcclient.InferInput("INPUT1", [1, 16], "INT32"),
        ]
        inputs[0].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))
        inputs[1].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))

        with self.assertRaisesRegex(
            InferenceServerException, "This protocol is restricted"
        ):
            _ = self.client_.infer(
                model_name=self.model_name_, inputs=inputs, headers={"test": "1"}
            )
        self.client_.infer(
            model_name=self.model_name_,
            inputs=inputs,
            headers={self.prefix_ + "infer-key": "infer-value"},
        )

    def test_stream_infer(self):

        inputs = [
            grpcclient.InferInput("INPUT0", [1, 16], "INT32"),
            grpcclient.InferInput("INPUT1", [1, 16], "INT32"),
        ]
        inputs[0].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))
        inputs[1].set_data_from_numpy(np.ones(shape=(1, 16), dtype=np.int32))
        user_data = UserData()

        self.client_.start_stream(partial(callback, user_data), headers={"test": "1"})

        time.sleep(1)
        with self.assertRaisesRegex(
            InferenceServerException, "The stream is no longer in valid state"
        ):
            self.client_.async_stream_infer(model_name=self.model_name_, inputs=inputs)

        self.assertFalse(user_data._completed_requests.empty())
        with self.assertRaisesRegex(
            InferenceServerException, "This protocol is restricted"
        ):
            raise user_data._completed_requests.get()

        self.assertTrue(user_data._completed_requests.empty())

        self.client_.stop_stream()
        self.client_.start_stream(
            partial(callback, user_data),
            headers={self.prefix_ + "infer-key": "infer-value"},
        )
        self.client_.async_stream_infer(model_name=self.model_name_, inputs=inputs)

        time.sleep(1)
        self.assertFalse(user_data._completed_requests.empty())
        self.assertNotEqual(
            type(user_data._completed_requests.get()), InferenceServerException
        )


if __name__ == "__main__":
    unittest.main()
