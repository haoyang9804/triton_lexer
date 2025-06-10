import os
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class ResponseSenderTest(unittest.TestCase):
    def _generate_streaming_callback_and_responses_pair(self):
        responses = []

        def callback(result, error):
            responses.append({"result": result, "error": error})

        return callback, responses

    def test_respond_after_complete_final(self):
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("Test Passed", server_log)

        model_name = "response_sender_complete_final"
        shape = [1, 1]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        input0_np = np.array([[123.45]], np.float32)
        inputs[0].set_data_from_numpy(input0_np)

        callback, responses = self._generate_streaming_callback_and_responses_pair()
        with grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001") as client:
            client.start_stream(callback)
            client.async_stream_infer(model_name, inputs)
            client.stop_stream()

        self.assertEqual(len(responses), 1)
        for response in responses:
            output0_np = response["result"].as_numpy(name="OUTPUT0")
            self.assertTrue(np.allclose(input0_np, output0_np))
            self.assertIsNone(response["error"])

        time.sleep(1)
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("Unexpected request length", server_log)
        self.assertNotIn("Expected exception not raised", server_log)
        self.assertNotIn("Test FAILED", server_log)
        self.assertIn("Test Passed", server_log)


if __name__ == "__main__":
    unittest.main()
