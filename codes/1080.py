import sys

sys.path.append("../common")

import functools
import time
import unittest
from builtins import range

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class ParallelCopyTest(tu.TestResultCollector):
    def setUp(self):
        self.client_ = grpcclient.InferenceServerClient("localhost:8001")
        self.dtype_ = np.float32
        self.model_name_ = tu.get_zero_model_name("plan", 1, self.dtype_)

    def _batch_input_duration(self, batch_size):
        stats = self.client_.get_inference_statistics(self.model_name_, "1")
        self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")
        self.assertEqual(
            stats.model_stats[0].name,
            self.model_name_,
            "expect model stats for model {}".format(self.model_name_),
        )
        self.assertEqual(
            stats.model_stats[0].version,
            "1",
            "expect model stats for model {} version 1".format(self.model_name_),
        )

        batch_stats = stats.model_stats[0].batch_stats

        batch_input_duration = 0
        for batch_stat in batch_stats:
            if batch_stat.batch_size == batch_size:
                batch_input_duration = batch_stat.compute_input.ns
        return batch_input_duration

    def _run(self, batch_sizes):
        batch_size = functools.reduce(lambda a, b: a + b, batch_sizes, 0)
        input_data = [
            np.random.random([bs, 16 * 1024 * 1024]).astype(self.dtype_)
            for bs in batch_sizes
        ]
        inputs = [
            [grpcclient.InferInput("INPUT0", [bs, 16 * 1024 * 1024], "FP32")]
            for bs in batch_sizes
        ]
        output = [grpcclient.InferRequestedOutput("OUTPUT0")]

        for idx in range(len(inputs)):
            inputs[idx][0].set_data_from_numpy(input_data[idx])

        def callback(user_data, idx, result, error):
            if error:
                user_data[idx] = error
            else:
                user_data[idx] = result

        user_data = [None] * len(batch_sizes)

        before_compute_input_duration = self._batch_input_duration(batch_size)
        for idx in range(len(batch_sizes)):
            self.client_.async_infer(
                model_name=self.model_name_,
                inputs=inputs[idx],
                callback=functools.partial(callback, user_data, idx),
                outputs=output,
            )

        time_out = 20
        while time_out > 0:
            done = True
            for res in user_data:
                if res is None:
                    done = False
                    break
            if done:
                break
            time_out = time_out - 1
            time.sleep(1)
        done_cnt = functools.reduce(
            lambda dc, x: dc + 1 if x is not None else dc, user_data, 0
        )
        self.assertEqual(
            done_cnt,
            len(batch_sizes),
            "expected {} responses, got {}".format(len(batch_sizes), done_cnt),
        )
        for idx in range(len(batch_sizes)):
            res = user_data[idx]
            self.assertFalse(
                type(res) == InferenceServerException,
                "expected response for request {}, got exception {}".format(idx, res),
            )
            output_data = res.as_numpy("OUTPUT0")
            self.assertTrue(
                np.array_equal(output_data, input_data[idx]),
                "Mismatched output data for request {}".format(idx),
            )

        after_compute_input_duration = self._batch_input_duration(batch_size)
        return after_compute_input_duration - before_compute_input_duration

    def test_performance(self):
        model_status = self.client_.is_model_ready(self.model_name_, "1")
        self.assertTrue(model_status, "expected model to be ready")

        serialized_time = self._run([8])
        parallelized_time = self._run([2, 2, 2, 2])

        self.assertTrue(
            serialized_time > parallelized_time,
            "Expected parallelized copy is faster than serialized copy",
        )
        print(
            "serialized v.s. parallelized : {} v.s. {}".format(
                serialized_time, parallelized_time
            )
        )


if __name__ == "__main__":
    unittest.main()
