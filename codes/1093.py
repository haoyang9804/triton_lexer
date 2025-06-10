import os
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient


class TestImplicitState(unittest.TestCase):
    def _get_inputs(self, delay_itrs):
        shape = [1, 1]
        inputs = [grpcclient.InferInput("DELAY_ITRS__0", shape, "INT64")]
        inputs[0].set_data_from_numpy(np.array([[delay_itrs]], np.int64))
        return inputs

    def _generate_streaming_callback_and_response_pair(self):
        response = []

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    def _sequence_state_model_infer(self, num_reqs, seq_ids, delay_itrs, cancel_reqs):
        model_name = "sequence_state"
        callback, response = self._generate_streaming_callback_and_response_pair()
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            client.start_stream(callback)
            seq_start = True
            for req_id in range(num_reqs):
                for seq_id in seq_ids:
                    client.async_stream_infer(
                        model_name,
                        self._get_inputs(delay_itrs),
                        sequence_id=seq_id,
                        sequence_start=seq_start,
                    )
                    time.sleep(0.1)
                seq_start = False
            client.stop_stream(cancel_requests=cancel_reqs)
        return response

    def test_state_reset_after_cancel(self):
        sequence_timeout = 6

        num_reqs = 10
        response = self._sequence_state_model_infer(
            num_reqs, seq_ids=[1], delay_itrs=5000000, cancel_reqs=True
        )
        self.assertLess(
            len(response),
            num_reqs,
            "Precondition not met - sequence completed before cancellation",
        )

        time.sleep(sequence_timeout + 2)

        self._sequence_state_model_infer(
            num_reqs=4, seq_ids=[2, 3], delay_itrs=0, cancel_reqs=False
        )

        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("[MODEL ERROR] Invalid sequence state", server_log)


if __name__ == "__main__":
    unittest.main()
