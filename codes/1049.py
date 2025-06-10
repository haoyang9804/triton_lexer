


























import sys

sys.path.append("../common")

import json


import queue
import unittest
from functools import partial

import numpy as np
import requests
import sseclient
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

MODEL_CONFIG_BASE = 


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class IterativeSequenceTest(tu.TestResultCollector):
    def setUp(self):
        
        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            triton_client.load_model("iterative_sequence")

    def test_generate_stream(self):
        headers = {"Accept": "text/event-stream"}
        url = "http://localhost:8000/v2/models/iterative_sequence/generate_stream"
        inputs = {"INPUT": 2}
        res = requests.post(url, data=json.dumps(inputs), headers=headers)
        res.raise_for_status()
        client = sseclient.SSEClient(res)
        res_count = 2
        for event in client.events():
            res_count -= 1
            data = json.loads(event.data)
            self.assertIn("OUTPUT", data)
            self.assertEqual(res_count, data["OUTPUT"])
        self.assertEqual(0, res_count)

    def test_grpc_stream(
        self, sequence_id=0, sequence_start=False, num_requests=1, validation=True
    ):
        user_data = UserData()
        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))
            inputs = []
            inputs.append(grpcclient.InferInput("INPUT", [1, 1], "INT32"))
            inputs[0].set_data_from_numpy(np.array([[2]], dtype=np.int32))

            for _ in range(num_requests):
                triton_client.async_stream_infer(
                    model_name="iterative_sequence",
                    inputs=inputs,
                    sequence_id=sequence_id,
                    sequence_start=sequence_start,
                )
            res_count = 2 * num_requests
            while res_count > 0:
                data_item = user_data._completed_requests.get()
                res_count -= 1
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    if validation:
                        self.assertEqual(
                            res_count % 2, data_item.as_numpy("OUTPUT")[0][0]
                        )
            self.assertEqual(0, res_count)

    def test_backlog_fill(self):
        config = r'"sequence_batching" : { "iterative_sequence" : true, "max_sequence_idle_microseconds": 8000000, direct: { "max_queue_delay_microseconds" : 10000000 }}'
        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            triton_client.load_model(
                "iterative_sequence", config=MODEL_CONFIG_BASE.format(config)
            )
        self.test_grpc_stream(num_requests=4, validation=False)

    def test_reschedule_error(self):
        
        
        
        config = r'"sequence_batching" : { "iterative_sequence" : true, "max_sequence_idle_microseconds" : 200000 }'
        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            triton_client.load_model(
                "iterative_sequence", config=MODEL_CONFIG_BASE.format(config)
            )
        with self.assertRaises(InferenceServerException) as context:
            
            
            self.test_grpc_stream()
        print(str(context.exception))
        self.assertTrue(
            "must specify the START flag on the first request of the sequence"
            in str(context.exception)
        )

    def test_unsupported_sequence_scheduler(self):
        
        
        configs = [
            r'"sequence_batching" : { "direct" : {}, "iterative_sequence" : false }',
            r'"sequence_batching" : { "oldest" : {}, "iterative_sequence" : false }',
        ]
        sid = 1
        for sc in configs:
            with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
                triton_client.load_model(
                    "iterative_sequence", config=MODEL_CONFIG_BASE.format(sc)
                )
            with self.assertRaises(InferenceServerException) as context:
                
                
                self.test_grpc_stream(sequence_id=sid, sequence_start=True)
            sid += 1
            self.assertTrue(
                "Request is released with TRITONSERVER_REQUEST_RELEASE_RESCHEDULE"
                in str(context.exception)
            )

    def test_unsupported_dynamic_scheduler(self):
        
        
        configs = [
            r'"dynamic_batching" : {}',
        ]
        for sc in configs:
            with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
                triton_client.load_model(
                    "iterative_sequence", config=MODEL_CONFIG_BASE.format(sc)
                )
            with self.assertRaises(InferenceServerException) as context:
                self.test_grpc_stream()
            self.assertTrue(
                "Request is released with TRITONSERVER_REQUEST_RELEASE_RESCHEDULE"
                in str(context.exception)
            )


if __name__ == "__main__":
    unittest.main()
