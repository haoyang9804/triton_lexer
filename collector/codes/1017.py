import sys

sys.path.append("../common")

import os
import threading
import time
import unittest
from builtins import range

import infer_util as iu
import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

TEST_SYSTEM_SHARED_MEMORY = bool(int(os.environ.get("TEST_SYSTEM_SHARED_MEMORY", 0)))
TEST_CUDA_SHARED_MEMORY = bool(int(os.environ.get("TEST_CUDA_SHARED_MEMORY", 0)))

if TEST_SYSTEM_SHARED_MEMORY:
    import tritonclient.utils.shared_memory as shm
if TEST_CUDA_SHARED_MEMORY:
    import tritonclient.utils.cuda_shared_memory as cudashm


USE_GRPC = os.environ.get("USE_GRPC", 1) != "0"
USE_HTTP = os.environ.get("USE_HTTP", 1) != "0"
if USE_GRPC and USE_HTTP:
    USE_GRPC = False
assert USE_GRPC or USE_HTTP, "USE_GRPC or USE_HTTP must be non-zero"

BACKENDS = os.environ.get("BACKENDS", "onnx libtorch plan python")

_trials = BACKENDS.split(" ")

_ragged_batch_supported_trials = ["custom"]
if "plan" in _trials:
    _ragged_batch_supported_trials.append("plan")
if "onnx" in _trials:
    _ragged_batch_supported_trials.append("onnx")
if "libtorch" in _trials:
    _ragged_batch_supported_trials.append("libtorch")

_max_queue_delay_ms = 10000

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
        if TEST_SYSTEM_SHARED_MEMORY:
            self.triton_client_.unregister_system_shared_memory()
        if TEST_CUDA_SHARED_MEMORY:
            self.triton_client_.unregister_cuda_shared_memory()
        for precreated_shm_region in self.precreated_shm_regions_:
            if TEST_SYSTEM_SHARED_MEMORY:
                shm.destroy_shared_memory_region(precreated_shm_region)
            elif TEST_CUDA_SHARED_MEMORY:
                cudashm.destroy_shared_memory_region(precreated_shm_region)
        super().tearDown()

    def create_advance(self, shm_regions=None):
        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            precreated_shm_regions = []
            if shm_regions is None:
                shm_regions = ["output0", "output1"]
            for shm_region in shm_regions:
                if TEST_SYSTEM_SHARED_MEMORY:
                    shm_handle = shm.create_shared_memory_region(
                        shm_region + "_data", "/" + shm_region, 512
                    )
                    self.triton_client_.register_system_shared_memory(
                        shm_region + "_data", "/" + shm_region, 512
                    )
                else:
                    shm_handle = cudashm.create_shared_memory_region(
                        shm_region + "_data", 512, 0
                    )
                    self.triton_client_.register_cuda_shared_memory(
                        shm_region + "_data", cudashm.get_raw_handle(shm_handle), 0, 512
                    )

                self.precreated_shm_regions_.append(shm_handle)
                precreated_shm_regions.append(shm_handle)
            return precreated_shm_regions
        return []

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
                    use_http_json_tensors=False,
                    use_grpc=USE_GRPC,
                    use_http=USE_HTTP,
                    skip_request_id_check=True,
                    use_streaming=False,
                    shm_region_names=shm_region_names,
                    precreated_shm_regions=precreated_shm_regions,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY,
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

    def check_setup(self, model_name, preferred_batch_sizes, max_queue_delay_us):

        config = self.triton_client_.get_model_config(model_name).config
        bconfig = config.dynamic_batching
        self.assertEqual(len(bconfig.preferred_batch_size), len(preferred_batch_sizes))
        for i in preferred_batch_sizes:
            self.assertTrue(i in bconfig.preferred_batch_size)
        self.assertEqual(bconfig.max_queue_delay_microseconds, max_queue_delay_us)

    def check_status(self, model_name, batch_exec, request_cnt, infer_cnt, exec_count):

        num_tries = 10
        for i in range(num_tries):
            stats = self.triton_client_.get_inference_statistics(model_name, "1")
            self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")
            actual_exec_cnt = stats.model_stats[0].execution_count
            if actual_exec_cnt in exec_count:
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
        self.assertIn(
            actual_exec_cnt,
            exec_count,
            "expected model-exec-count {}, got {}".format(exec_count, actual_exec_cnt),
        )

        actual_infer_cnt = stats.model_stats[0].inference_count
        self.assertEqual(
            actual_infer_cnt,
            infer_cnt,
            "expected model-inference-count {}, got {}".format(
                infer_cnt, actual_infer_cnt
            ),
        )

    def test_static_batch_preferred(self):

        precreated_shm_regions = self.create_advance()
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(
                    trial,
                    2,
                    (3000, None),
                    precreated_shm_regions=precreated_shm_regions,
                )
                self.check_response(
                    trial,
                    6,
                    (3000, None),
                    precreated_shm_regions=precreated_shm_regions,
                )
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1, 6: 1}, 2, 8, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_lt_any_preferred(self):

        precreated_shm_regions = self.create_advance()
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(
                    trial,
                    1,
                    (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                    precreated_shm_regions=precreated_shm_regions,
                )
                self.check_deferred_exception()
                self.check_status(model_name, {1: 1}, 1, 1, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_not_preferred(self):

        precreated_shm_regions = self.create_advance()
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(
                    trial,
                    3,
                    (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                    precreated_shm_regions=precreated_shm_regions,
                )
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1}, 1, 3, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_static_batch_gt_max_preferred(self):

        precreated_shm_regions = self.create_advance()
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                self.check_response(
                    trial,
                    7,
                    (3000, None),
                    precreated_shm_regions=precreated_shm_regions,
                )
                self.check_deferred_exception()
                self.check_status(model_name, {7: 1}, 1, 7, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_different_shape_allow_ragged(self):

        for trial in _ragged_batch_supported_trials:
            try:
                dtype = np.float32
                model_name = tu.get_zero_model_name(trial, 1, dtype)

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=iu.infer_zero,
                        args=(self, trial, 1, dtype, ([1, 16],), ([1, 16],)),
                        kwargs={
                            "use_grpc": USE_GRPC,
                            "use_http": USE_HTTP,
                            "use_http_json_tensors": False,
                            "use_streaming": False,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=iu.infer_zero,
                        args=(self, trial, 1, dtype, ([1, 8],), ([1, 8],)),
                        kwargs={
                            "use_grpc": USE_GRPC,
                            "use_http": USE_HTTP,
                            "use_http_json_tensors": False,
                            "use_streaming": False,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_different_shape(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "input_size": 16,
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            1,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                        ),
                        kwargs={
                            "input_size": 8,
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {1: 2}, 2, 2, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_not_preferred(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            1,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                        ),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            3,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms - 2000),
                        ),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1}, 2, 4, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_not_preferred_different_shape(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (6000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            1,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                        ),
                        kwargs={
                            "input_size": 8,
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {1: 1, 4: 1}, 3, 5, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_preferred_different_shape(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
            shm3_region_names = ["ip30", "ip31", "op30", "op31"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        precreated_shm3_regions = self.create_advance(["op30", "op31"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (6000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "input_size": 8,
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 5, (6000, None)),
                        kwargs={
                            "input_size": 8,
                            "shm_region_names": shm3_region_names,
                            "precreated_shm_regions": precreated_shm3_regions,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                threads[3].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1, 6: 1}, 4, 10, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_gt_max_preferred(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (3000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 7, (3000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1, 7: 1}, 2, 10, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_sum_gt_max_preferred(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)
                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (3000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            4,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                        ),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1, 4: 1}, 2, 7, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_same_output0(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00"]
            shm1_region_names = ["ip10", "ip11", "op10"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(["op00"])
        precreated_shm1_regions = self.create_advance(["op10"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            "requested_outputs": ("OUTPUT0",),
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            "requested_outputs": ("OUTPUT0",),
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_same_output1(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op01"]
            shm1_region_names = ["ip10", "ip11", "op11"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(["op01"])
        precreated_shm1_regions = self.create_advance(["op11"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            "requested_outputs": ("OUTPUT1",),
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            "requested_outputs": ("OUTPUT1",),
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_different_outputs(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00"]
            shm1_region_names = ["ip10", "ip11", "op11"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(["op00"])
        precreated_shm1_regions = self.create_advance(["op11"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "requested_outputs": ("OUTPUT0",),
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "requested_outputs": ("OUTPUT1",),
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_different_output_order(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op11", "op10"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertFalse("TRITONSERVER_DELAY_SCHEDULER" in os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "requested_outputs": ("OUTPUT0", "OUTPUT1"),
                            "shm_region_names": shm0_region_names,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "requested_outputs": ("OUTPUT1", "OUTPUT0"),
                            "shm_region_names": shm1_region_names,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1}, 2, 2, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_delayed_sum_gt_max_preferred(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
        else:
            shm0_region_names = None
            shm1_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 2)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            4,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                        ),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1, 4: 1}, 2, 7, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_delayed_use_max_batch(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 3)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            3,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                        ),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            4,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                        ),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                time.sleep(11)
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {8: 1}, 3, 8, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_delayed_preferred_different_shape(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
            shm3_region_names = ["ip30", "ip31", "op30", "op31"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        precreated_shm3_regions = self.create_advance(["op30", "op31"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 4)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 3, (3000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (3000, None)),
                        kwargs={
                            "input_size": 8,
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 5, (3000, None)),
                        kwargs={
                            "input_size": 8,
                            "shm_region_names": shm3_region_names,
                            "precreated_shm_regions": precreated_shm3_regions,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                threads[3].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1, 6: 1}, 4, 10, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_use_biggest_preferred(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
            shm3_region_names = ["ip30", "ip31", "op30", "op31"]
            shm4_region_names = ["ip40", "ip41", "op40", "op41"]
            shm5_region_names = ["ip50", "ip51", "op50", "op51"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
            shm4_region_names = None
            shm5_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        precreated_shm3_regions = self.create_advance(["op30", "op31"])
        precreated_shm4_regions = self.create_advance(["op40", "op41"])
        precreated_shm5_regions = self.create_advance(["op50", "op51"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 6)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm3_region_names,
                            "precreated_shm_regions": precreated_shm3_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm4_region_names,
                            "precreated_shm_regions": precreated_shm4_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm5_region_names,
                            "precreated_shm_regions": precreated_shm5_regions,
                        },
                    )
                )
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {6: 1}, 6, 6, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_use_best_preferred(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [2, 6], _max_queue_delay_ms * 1000)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 3)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(
                            trial,
                            1,
                            (_max_queue_delay_ms * 1.5, _max_queue_delay_ms),
                        ),
                        kwargs={
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads[0].start()
                threads[1].start()
                time.sleep(1)
                threads[2].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {2: 1, 1: 1}, 3, 3, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_multi_batch_preserve_ordering(self):
        model_base = "custom"
        dtype = np.float32
        shapes = (
            [
                1,
                1,
            ],
        )

        try:

            threads = []
            for i in range(12):
                if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                    shm_region_name_prefix = ["input" + str(i), "output" + str(i)]
                else:
                    shm_region_name_prefix = None
                threads.append(
                    threading.Thread(
                        target=iu.infer_zero,
                        args=(self, model_base, 1, dtype, shapes, shapes),
                        kwargs={
                            "use_grpc": USE_GRPC,
                            "use_http": USE_HTTP,
                            "use_http_json_tensors": False,
                            "use_streaming": False,
                            "shm_region_name_prefix": shm_region_name_prefix,
                            "use_system_shared_memory": TEST_SYSTEM_SHARED_MEMORY,
                            "use_cuda_shared_memory": TEST_CUDA_SHARED_MEMORY,
                        },
                    )
                )
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            model_name = tu.get_zero_model_name(model_base, len(shapes), dtype)
            self.check_status(model_name, {4: 3}, 12, 12, (3,))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_preferred_batch_only_aligned(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
            shm3_region_names = ["ip30", "ip31", "op30", "op31"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        precreated_shm3_regions = self.create_advance(["op30", "op31"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [4, 6], 0)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 4)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm3_region_names,
                            "precreated_shm_regions": precreated_shm3_regions,
                        },
                    )
                )
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1}, 4, 4, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_preferred_batch_only_unaligned(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
            shm3_region_names = ["ip30", "ip31", "op30", "op31"]
            shm4_region_names = ["ip40", "ip41", "op40", "op41"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
            shm4_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        precreated_shm3_regions = self.create_advance(["op30", "op31"])
        precreated_shm4_regions = self.create_advance(["op40", "op41"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [4, 6], 0)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 5)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm3_region_names,
                            "precreated_shm_regions": precreated_shm3_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm4_region_names,
                            "precreated_shm_regions": precreated_shm4_regions,
                        },
                    )
                )
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 1, 1: 1}, 5, 5, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_preferred_batch_only_use_biggest_preferred(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
            shm3_region_names = ["ip30", "ip31", "op30", "op31"]
            shm4_region_names = ["ip40", "ip41", "op40", "op41"]
            shm5_region_names = ["ip50", "ip51", "op50", "op51"]
            shm6_region_names = ["ip60", "ip61", "op60", "op61"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
            shm3_region_names = None
            shm4_region_names = None
            shm5_region_names = None
            shm6_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        precreated_shm3_regions = self.create_advance(["op30", "op31"])
        precreated_shm4_regions = self.create_advance(["op40", "op41"])
        precreated_shm5_regions = self.create_advance(["op50", "op51"])
        precreated_shm6_regions = self.create_advance(["op60", "op61"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [4, 6], 0)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 7)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm3_region_names,
                            "precreated_shm_regions": precreated_shm3_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm4_region_names,
                            "precreated_shm_regions": precreated_shm4_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm5_region_names,
                            "precreated_shm_regions": precreated_shm5_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm6_region_names,
                            "precreated_shm_regions": precreated_shm6_regions,
                        },
                    )
                )
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {6: 1, 1: 1}, 7, 7, (2,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_preferred_batch_only_use_no_preferred_size(self):

        if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
            shm0_region_names = ["ip00", "ip01", "op00", "op01"]
            shm1_region_names = ["ip10", "ip11", "op10", "op11"]
            shm2_region_names = ["ip20", "ip21", "op20", "op21"]
        else:
            shm0_region_names = None
            shm1_region_names = None
            shm2_region_names = None
        precreated_shm0_regions = self.create_advance(["op00", "op01"])
        precreated_shm1_regions = self.create_advance(["op10", "op11"])
        precreated_shm2_regions = self.create_advance(["op20", "op21"])
        for trial in _trials:
            try:
                model_name = tu.get_model_name(
                    trial, np.float32, np.float32, np.float32
                )

                self.check_setup(model_name, [4, 6], 0)

                self.assertTrue("TRITONSERVER_DELAY_SCHEDULER" in os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 3)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm0_region_names,
                            "precreated_shm_regions": precreated_shm0_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm1_region_names,
                            "precreated_shm_regions": precreated_shm1_regions,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(trial, 1, (6000, None)),
                        kwargs={
                            "shm_region_names": shm2_region_names,
                            "precreated_shm_regions": precreated_shm2_regions,
                        },
                    )
                )
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 1}, 3, 3, (1,))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_max_queue_delay_only_non_default(self):

        model_base = "custom"
        dtype = np.float32
        shapes = (
            [
                1,
                1,
            ],
        )

        try:

            threads = []
            for i in range(12):
                if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                    shm_region_name_prefix = ["input" + str(i), "output" + str(i)]
                else:
                    shm_region_name_prefix = None
                threads.append(
                    threading.Thread(
                        target=iu.infer_zero,
                        args=(self, model_base, 1, dtype, shapes, shapes),
                        kwargs={
                            "use_grpc": USE_GRPC,
                            "use_http": USE_HTTP,
                            "use_http_json_tensors": False,
                            "use_streaming": False,
                            "shm_region_name_prefix": shm_region_name_prefix,
                            "use_system_shared_memory": TEST_SYSTEM_SHARED_MEMORY,
                            "use_cuda_shared_memory": TEST_CUDA_SHARED_MEMORY,
                        },
                    )
                )
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            model_name = tu.get_zero_model_name(model_base, len(shapes), dtype)
            self.check_status(model_name, None, 12, 12, (1, 2))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_max_queue_delay_only_default(self):

        model_base = "custom"
        dtype = np.float32
        shapes = (
            [
                1,
                1,
            ],
        )

        try:

            threads = []
            for i in range(12):
                if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                    shm_region_name_prefix = ["input" + str(i), "output" + str(i)]
                else:
                    shm_region_name_prefix = None
                threads.append(
                    threading.Thread(
                        target=iu.infer_zero,
                        args=(self, model_base, 1, dtype, shapes, shapes),
                        kwargs={
                            "use_grpc": USE_GRPC,
                            "use_http": USE_HTTP,
                            "use_http_json_tensors": False,
                            "use_streaming": False,
                            "shm_region_name_prefix": shm_region_name_prefix,
                            "use_system_shared_memory": TEST_SYSTEM_SHARED_MEMORY,
                            "use_cuda_shared_memory": TEST_CUDA_SHARED_MEMORY,
                        },
                    )
                )
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.check_deferred_exception()
            model_name = tu.get_zero_model_name(model_base, len(shapes), dtype)
            self.check_status(model_name, None, 12, 12, (2,))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == "__main__":
    unittest.main()
