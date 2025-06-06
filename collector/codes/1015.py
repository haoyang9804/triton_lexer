import sys

sys.path.append("../common")

import os
import threading
import time
import unittest
from builtins import range
from collections.abc import Iterable

import infer_util as iu
import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


class BatcherTest(tu.TestResultCollector):
    def setUp(self):

        self.triton_client_ = grpcclient.InferenceServerClient(
            f"{_tritonserver_ipaddr}:8001"
        )
        self.precreated_shm_regions_ = []
        global _deferred_exceptions
        _deferred_exceptions = []

    def tearDown(self):
        super().tearDown()

    def add_deferred_exception(self, ex):
        global _deferred_exceptions
        with _deferred_exceptions_lock:
            _deferred_exceptions.append(ex)

    def check_deferred_exception(self):

        with _deferred_exceptions_lock:
            if len(_deferred_exceptions) > 0:
                raise _deferred_exceptions[0]

    def check_response(
        self,
        trial,
        bs,
        thresholds,
        requested_outputs=("OUTPUT0", "OUTPUT1"),
        input_size=16,
        shm_region_names=None,
        precreated_shm_regions=None,
    ):
        try:
            start_ms = int(round(time.time() * 1000))

            if (
                trial == "libtorch"
                or trial == "onnx"
                or trial == "plan"
                or trial == "python"
            ):
                tensor_shape = (bs, input_size)
                iu.infer_exact(
                    self,
                    trial,
                    tensor_shape,
                    bs,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=False,
                    model_version=1,
                    outputs=requested_outputs,
                    use_http=False,
                    use_grpc=False,
                    use_http_json_tensors=False,
                    skip_request_id_check=True,
                    use_streaming=False,
                )
            else:
                self.assertFalse(True, "unknown trial type: " + trial)

            end_ms = int(round(time.time() * 1000))

            lt_ms = thresholds[0]
            gt_ms = thresholds[1]
            if lt_ms is not None:
                self.assertTrue(
                    (end_ms - start_ms) < lt_ms,
                    "expected less than "
                    + str(lt_ms)
                    + "ms response time, got "
                    + str(end_ms - start_ms)
                    + " ms",
                )
            if gt_ms is not None:
                self.assertTrue(
                    (end_ms - start_ms) > gt_ms,
                    "expected greater than "
                    + str(gt_ms)
                    + "ms response time, got "
                    + str(end_ms - start_ms)
                    + " ms",
                )
        except Exception as ex:
            self.add_deferred_exception(ex)

    def check_status(self, model_name, batch_exec, request_cnt, infer_cnt, exec_count):

        num_tries = 10
        for i in range(num_tries):
            stats = self.triton_client_.get_inference_statistics(model_name, "1")
            self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")
            actual_exec_cnt = stats.model_stats[0].execution_count
            if actual_exec_cnt == exec_count:
                break
            print(
                "WARNING: expect {} executions, got {} (attempt {})".format(
                    exec_count, actual_exec_cnt, i
                )
            )
            time.sleep(1)

        self.assertEqual(
            stats.model_stats[0].name,
            model_name,
            "expect model stats for model {}".format(model_name),
        )
        self.assertEqual(
            stats.model_stats[0].version,
            "1",
            "expect model stats for model {} version 1".format(model_name),
        )

        if batch_exec:
            batch_stats = stats.model_stats[0].batch_stats
            self.assertEqual(
                len(batch_stats),
                len(batch_exec),
                "expected {} different batch-sizes, got {}".format(
                    len(batch_exec), len(batch_stats)
                ),
            )

            for batch_stat in batch_stats:
                bs = batch_stat.batch_size
                bc = batch_stat.compute_infer.count
                self.assertTrue(bs in batch_exec, "unexpected batch-size {}".format(bs))

                self.assertEqual(
                    bc,
                    batch_exec[bs],
                    "expected model-execution-count {} for batch size {}, got {}".format(
                        batch_exec[bs], bs, bc
                    ),
                )

        actual_request_cnt = stats.model_stats[0].inference_stats.success.count
        self.assertEqual(
            actual_request_cnt,
            request_cnt,
            "expected model-request-count {}, got {}".format(
                request_cnt, actual_request_cnt
            ),
        )

        actual_exec_cnt = stats.model_stats[0].execution_count
        if isinstance(exec_count, Iterable):
            self.assertIn(
                actual_exec_cnt,
                exec_count,
                "expected model-exec-count {}, got {}".format(
                    exec_count, actual_exec_cnt
                ),
            )
        else:
            self.assertEqual(
                actual_exec_cnt,
                exec_count,
                "expected model-exec-count {}, got {}".format(
                    exec_count, actual_exec_cnt
                ),
            )
        actual_infer_cnt = stats.model_stats[0].inference_count
        self.assertEqual(
            actual_infer_cnt,
            infer_cnt,
            "expected model-inference-count {}, got {}".format(
                infer_cnt, actual_infer_cnt
            ),
        )

    def test_volume_batching(self):

        model_base = "onnx"
        dtype = np.float16
        shapes = (
            [
                1,
                4,
                4,
            ],
        )

        try:

            threads = []
            for i in range(12):
                threads.append(
                    threading.Thread(
                        target=iu.infer_zero,
                        args=(self, model_base, 1, dtype, shapes, shapes),
                        kwargs={
                            "use_http": True,
                            "use_grpc": False,
                            "use_http_json_tensors": False,
                            "use_streaming": False,
                        },
                    )
                )
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            model_name = tu.get_zero_model_name(model_base, len(shapes), dtype)
            self.check_status(model_name, None, 12, 12, (4, 5, 6))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == "__main__":
    unittest.main()
