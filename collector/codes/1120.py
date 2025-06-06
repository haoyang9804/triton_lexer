import sys

sys.path.append("../common")

import os
import threading
import time
import unittest
from builtins import range

import infer_util as iu
import numpy as np
import sequence_util as su
import test_util as tu
import tritonclient.grpc as grpcclient

TEST_SYSTEM_SHARED_MEMORY = bool(int(os.environ.get("TEST_SYSTEM_SHARED_MEMORY", 0)))

_model_instances = 1
_max_queue_delay_ms = 10000
_max_sequence_idle_ms = 5000

_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []


class InferShapeTensorTest(tu.TestResultCollector):
    def setUp(self):

        self.triton_client_ = grpcclient.InferenceServerClient("localhost:8001")
        global _deferred_exceptions
        _deferred_exceptions = []

    def tearDown(self):
        self.triton_client_.unregister_system_shared_memory()
        self.triton_client_.unregister_cuda_shared_memory()
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
        bs,
        thresholds,
        shape_values,
        dummy_input_shapes,
        shm_region_names=None,
        precreated_shm_regions=None,
        shm_suffix="",
        shape_tensor_input_dtype=np.int32,
    ):
        try:

            for i in range(len(dummy_input_shapes)):
                dummy_input_shapes[i] = [
                    bs,
                ] + dummy_input_shapes[i]
            start_ms = int(round(time.time() * 1000))

            iu.infer_shape_tensor(
                self,
                "plan",
                np.float32,
                shape_values,
                dummy_input_shapes,
                use_grpc=False,
                use_streaming=False,
                shm_suffix=shm_suffix,
                use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                batch_size=bs,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )

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

    def check_setup(self, model_name):

        config = self.triton_client_.get_model_config(model_name).config
        bconfig = config.dynamic_batching
        self.assertTrue(2 in bconfig.preferred_batch_size)
        self.assertTrue(6 in bconfig.preferred_batch_size)
        self.assertEqual(
            bconfig.max_queue_delay_microseconds, _max_queue_delay_ms * 1000
        )

    def check_status(self, model_name, batch_exec, exec_cnt, infer_cnt):

        num_tries = 10
        for i in range(num_tries):
            stats = self.triton_client_.get_inference_statistics(model_name, "1")
            self.assertEqual(len(stats.model_stats), 1, "expect 1 model stats")
            actual_exec_cnt = stats.model_stats[0].execution_count
            if actual_exec_cnt == exec_cnt:
                break
            print(
                "WARNING: expect {} executions, got {} (attempt {})".format(
                    exec_cnt, actual_exec_cnt, i
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

        if batch_exec is not None:
            batch_stats = stats.model_stats[0].batch_stats
            print(batch_stats)
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
                self.assertTrue(
                    bs in batch_exec, "did not find expected batch-size {}".format(bs)
                )

                self.assertEqual(
                    bc,
                    batch_exec[bs],
                    "expected model-execution-count {} for batch size {}, got {}".format(
                        batch_exec[bs], bs, bc
                    ),
                )

        actual_exec_cnt = stats.model_stats[0].execution_count
        self.assertEqual(
            actual_exec_cnt,
            exec_cnt,
            "expected model-exec-count {}, got {}".format(exec_cnt, actual_exec_cnt),
        )

        actual_infer_cnt = stats.model_stats[0].inference_count
        self.assertEqual(
            actual_infer_cnt,
            infer_cnt,
            "expected model-inference-count {}, got {}".format(
                infer_cnt, actual_infer_cnt
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

    def test_static_batch(self):
        for shape_tensor_input_dtype in [np.int32, np.int64]:
            iu.infer_shape_tensor(
                self,
                "plan",
                np.float32,
                [[32, 32]],
                [[8, 4, 4]],
                use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                batch_size=8,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )
            iu.infer_shape_tensor(
                self,
                "plan",
                np.float32,
                [[4, 4]],
                [[8, 32, 32]],
                use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                batch_size=8,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )
            iu.infer_shape_tensor(
                self,
                "plan",
                np.float32,
                [[4, 4]],
                [[8, 4, 4]],
                use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                batch_size=8,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )

    def test_nobatch(self):
        for shape_tensor_input_dtype in [np.int32, np.int64]:
            iu.infer_shape_tensor(
                self,
                "plan_nobatch",
                np.float32,
                [[32, 32]],
                [[4, 4]],
                use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )
            iu.infer_shape_tensor(
                self,
                "plan_nobatch",
                np.float32,
                [[4, 4]],
                [[32, 32]],
                use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )
            iu.infer_shape_tensor(
                self,
                "plan_nobatch",
                np.float32,
                [[4, 4]],
                [[4, 4]],
                use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )

    def test_wrong_shape_values(self):
        over_shape_values = [[32, 33]]
        for shape_tensor_input_dtype in [np.int32, np.int64]:
            try:
                iu.infer_shape_tensor(
                    self,
                    "plan",
                    np.float32,
                    over_shape_values,
                    [[8, 4, 4]],
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    batch_size=8,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )

            except Exception as ex:
                self.assertIn(
                    "The shape value at index 2 is expected to be in range from 1 to 32, Got: 33",
                    ex.message(),
                )

    def test_dynamic_different_shape_values(self):

        for shape_tensor_input_dtype in [np.int32, np.int64]:
            try:
                model_name = tu.get_zero_model_name("plan", 1, np.float32)
                model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name

                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(3, (6000, None)),
                        kwargs={
                            "shape_values": [[2, 2]],
                            "dummy_input_shapes": [[16, 16]],
                            "shm_suffix": "{}".format(len(threads)),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(3, (_max_queue_delay_ms * 1.5, _max_queue_delay_ms)),
                        kwargs={
                            "shape_values": [[4, 4]],
                            "dummy_input_shapes": [[16, 16]],
                            "shm_suffix": "{}".format(len(threads)),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {3: 2}, 2, 6)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_identical_shape_values(self):

        for shape_tensor_input_dtype in [np.int32, np.int64]:
            try:
                model_name = tu.get_zero_model_name("plan", 1, np.float32)
                model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name

                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(4, (6000, None)),
                        kwargs={
                            "shape_values": [[4, 4]],
                            "dummy_input_shapes": [[16, 16]],
                            "shm_suffix": "{}".format(len(threads)),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_response,
                        args=(2, (6000, None)),
                        kwargs={
                            "shape_values": [[4, 4]],
                            "dummy_input_shapes": [[16, 16]],
                            "shm_suffix": "{}".format(len(threads)),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads[0].start()
                time.sleep(1)
                threads[1].start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {6: 1}, 1, 6)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))


class SequenceBatcherShapeTensorTest(su.SequenceBatcherTestUtil):
    def get_expected_result(self, expected_result, value, flag_str=None):

        expected_result = value
        if (flag_str is not None) and ("start" in flag_str):
            expected_result += 1
        return expected_result

    def test_sequence_identical_shape_values(self):

        self.clear_deferred_exceptions()
        dtype = np.float32
        for shape_tensor_input_dtype in [np.int32, np.int64]:
            try:
                model_name = tu.get_sequence_model_name("plan", dtype)
                model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name
                self.check_setup(model_name)

                self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12)
                self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                )
                precreated_shm0_handles = self.precreate_register_shape_tensor_regions(
                    value_list=((2, 1), (4, 2), (8, 3)),
                    dtype=dtype,
                    i=0,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
                precreated_shm1_handles = self.precreate_register_shape_tensor_regions(
                    value_list=((2, 11), (4, 12), (8, 13)),
                    dtype=dtype,
                    i=1,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
                precreated_shm2_handles = self.precreate_register_shape_tensor_regions(
                    value_list=((2, 111), (4, 112), (8, 113)),
                    dtype=dtype,
                    i=2,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
                precreated_shm3_handles = self.precreate_register_shape_tensor_regions(
                    value_list=((2, 1111), (4, 1112), (8, 1113)),
                    dtype=dtype,
                    i=3,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            1001,
                            (None, None),
                            (
                                ("start", 2, 1, None),
                                (None, 4, 2, None),
                                ("end", 8, 3, None),
                            ),
                            self.get_expected_result(6, 3, "end"),
                            precreated_shm0_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}".format(self._testMethodName),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            1002,
                            (None, None),
                            (
                                ("start", 2, 11, None),
                                (None, 4, 12, None),
                                ("end", 8, 13, None),
                            ),
                            self.get_expected_result(36, 13, "end"),
                            precreated_shm1_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}".format(self._testMethodName),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            1003,
                            (None, None),
                            (
                                ("start", 2, 111, None),
                                (None, 4, 112, None),
                                ("end", 8, 113, None),
                            ),
                            self.get_expected_result(336, 113, "end"),
                            precreated_shm2_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}".format(self._testMethodName),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            1004,
                            (None, None),
                            (
                                ("start", 2, 1111, None),
                                (None, 4, 1112, None),
                                ("end", 8, 1113, None),
                            ),
                            self.get_expected_result(3336, 1113, "end"),
                            precreated_shm3_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}".format(self._testMethodName),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 3}, 3, 12)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if TEST_SYSTEM_SHARED_MEMORY:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)

    def test_sequence_different_shape_values(self):

        self.clear_deferred_exceptions()
        dtype = np.float32

        for shape_tensor_input_dtype in [np.int32, np.int64]:
            precreated_shm0_handles = self.precreate_register_shape_tensor_regions(
                value_list=((1, 1), (1, 2), (1, 3)),
                dtype=dtype,
                i=0,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )
            precreated_shm1_handles = self.precreate_register_shape_tensor_regions(
                value_list=((32, 11), (32, 12), (32, 13)),
                dtype=dtype,
                i=1,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )
            precreated_shm2_handles = self.precreate_register_shape_tensor_regions(
                value_list=((16, 111), (16, 112), (16, 113)),
                dtype=dtype,
                i=2,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )
            precreated_shm3_handles = self.precreate_register_shape_tensor_regions(
                value_list=((1, 1111), (1, 1112), (1, 1113)),
                dtype=dtype,
                i=3,
                shape_tensor_input_dtype=shape_tensor_input_dtype,
            )
            try:
                model_name = tu.get_sequence_model_name("plan", dtype)
                model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name
                self.check_setup(model_name)

                self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12)
                self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                self.assertEqual(
                    int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                )

                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            1001,
                            (None, None),
                            (
                                ("start", 1, 1, None),
                                (None, 1, 2, None),
                                ("end", 1, 3, None),
                            ),
                            self.get_expected_result(6, 3, "end"),
                            precreated_shm0_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}".format(self._testMethodName),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            1002,
                            (None, None),
                            (
                                ("start", 32, 11, None),
                                (None, 32, 12, None),
                                ("end", 32, 13, None),
                            ),
                            self.get_expected_result(36, 13, "end"),
                            precreated_shm1_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}".format(self._testMethodName),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            1003,
                            (None, None),
                            (
                                ("start", 16, 111, None),
                                (None, 16, 112, None),
                                ("end", 16, 113, None),
                            ),
                            self.get_expected_result(336, 113, "end"),
                            precreated_shm2_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}".format(self._testMethodName),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            1004,
                            (None, None),
                            (
                                ("start", 1, 1111, None),
                                (None, 1, 1112, None),
                                ("end", 1, 1113, None),
                            ),
                            self.get_expected_result(3336, 1113, "end"),
                            precreated_shm3_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}".format(self._testMethodName),
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )

                for t in threads:
                    t.start()
                    time.sleep(1)
                for t in threads:
                    t.join()

                self.check_deferred_exception()
                self.check_status(model_name, {4: 3, 3: 6}, 9, 12)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if TEST_SYSTEM_SHARED_MEMORY:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)


class DynaSequenceBatcherTest(su.SequenceBatcherTestUtil):
    def get_expected_result(self, expected_result, corrid, value, flag_str=None):
        expected_result = value
        if flag_str is not None:
            if "start" in flag_str:
                expected_result += 1
            if "end" in flag_str:
                expected_result += corrid
        return expected_result

    def _multi_sequence_different_shape_impl(self, sleep_secs):
        self.clear_deferred_exceptions()
        dtype = np.float32

        for shape_tensor_input_dtype in [np.int32, np.int64]:
            precreated_shm0_handles = (
                self.precreate_register_dynaseq_shape_tensor_regions(
                    value_list=((1, 1), (12, 2), (2, 3)),
                    dtype=dtype,
                    i=0,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
            )
            precreated_shm1_handles = (
                self.precreate_register_dynaseq_shape_tensor_regions(
                    value_list=((3, 11), (4, 12), (5, 13)),
                    dtype=dtype,
                    i=1,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
            )
            precreated_shm2_handles = (
                self.precreate_register_dynaseq_shape_tensor_regions(
                    value_list=((6, 111), (7, 112), (8, 113)),
                    dtype=dtype,
                    i=2,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
            )
            precreated_shm3_handles = (
                self.precreate_register_dynaseq_shape_tensor_regions(
                    value_list=((9, 1111), (10, 1112), (11, 1113)),
                    dtype=dtype,
                    i=3,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
            )

            try:
                model_name = tu.get_dyna_sequence_model_name("plan", dtype)
                model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name
                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)

                corrids = [1001, 1002, 1003, 1004]
                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            corrids[0],
                            (None, None),
                            (
                                ("start", 1, 1, None),
                                (None, 12, 2, None),
                                ("end", 2, 3, None),
                            ),
                            self.get_expected_result(
                                4 + corrids[0], corrids[0], 3, "end"
                            ),
                            precreated_shm0_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}_{}".format(
                                self._testMethodName, corrids[0]
                            ),
                            "using_dynamic_batcher": True,
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            corrids[1],
                            (None, None),
                            (
                                ("start", 3, 11, None),
                                (None, 4, 12, None),
                                ("end", 5, 13, None),
                            ),
                            self.get_expected_result(
                                36 + corrids[1], corrids[1], 13, "end"
                            ),
                            precreated_shm1_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}_{}".format(
                                self._testMethodName, corrids[1]
                            ),
                            "using_dynamic_batcher": True,
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            corrids[2],
                            (None, None),
                            (
                                ("start", 6, 111, None),
                                (None, 7, 112, None),
                                ("end", 8, 113, None),
                            ),
                            self.get_expected_result(
                                336 + corrids[2], corrids[2], 113, "end"
                            ),
                            precreated_shm2_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}_{}".format(
                                self._testMethodName, corrids[2]
                            ),
                            "using_dynamic_batcher": True,
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            corrids[3],
                            (None, None),
                            (
                                ("start", 9, 1111, None),
                                (None, 10, 1112, None),
                                ("end", 11, 1113, None),
                            ),
                            self.get_expected_result(
                                3336 + corrids[3], corrids[3], 1113, "end"
                            ),
                            precreated_shm3_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}_{}".format(
                                self._testMethodName, corrids[3]
                            ),
                            "using_dynamic_batcher": True,
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )

                for t in threads:
                    t.start()
                    if sleep_secs > 0:
                        time.sleep(sleep_secs)
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {1: 12}, 12, 12)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if TEST_SYSTEM_SHARED_MEMORY:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)

    def _multi_sequence_identical_shape_impl(self, sleep_secs):
        self.clear_deferred_exceptions()
        dtype = np.float32

        for shape_tensor_input_dtype in [np.int32, np.int64]:
            precreated_shm0_handles = (
                self.precreate_register_dynaseq_shape_tensor_regions(
                    value_list=((2, 1), (4, 2), (8, 3)),
                    dtype=dtype,
                    i=0,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
            )
            precreated_shm1_handles = (
                self.precreate_register_dynaseq_shape_tensor_regions(
                    value_list=((2, 11), (4, 12), (8, 13)),
                    dtype=dtype,
                    i=1,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
            )
            precreated_shm2_handles = (
                self.precreate_register_dynaseq_shape_tensor_regions(
                    value_list=((2, 111), (4, 112), (8, 113)),
                    dtype=dtype,
                    i=2,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
            )
            precreated_shm3_handles = (
                self.precreate_register_dynaseq_shape_tensor_regions(
                    value_list=((2, 1111), (4, 1112), (8, 1113)),
                    dtype=dtype,
                    i=3,
                    shape_tensor_input_dtype=shape_tensor_input_dtype,
                )
            )

            try:
                model_name = tu.get_dyna_sequence_model_name("plan", dtype)
                model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name
                self.check_setup(model_name)
                self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                self.assertNotIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)

                corrids = [1001, 1002, 1003, 1004]
                threads = []
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            corrids[0],
                            (None, None),
                            (
                                ("start", 2, 1, None),
                                (None, 4, 2, None),
                                ("end", 8, 3, None),
                            ),
                            self.get_expected_result(
                                4 + corrids[0], corrids[0], 3, "end"
                            ),
                            precreated_shm0_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}_{}".format(
                                self._testMethodName, corrids[0]
                            ),
                            "using_dynamic_batcher": True,
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            corrids[1],
                            (None, None),
                            (
                                ("start", 2, 11, None),
                                (None, 4, 12, None),
                                ("end", 8, 13, None),
                            ),
                            self.get_expected_result(
                                36 + corrids[1], corrids[1], 13, "end"
                            ),
                            precreated_shm1_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}_{}".format(
                                self._testMethodName, corrids[1]
                            ),
                            "using_dynamic_batcher": True,
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            corrids[2],
                            (None, None),
                            (
                                ("start", 2, 111, None),
                                (None, 4, 112, None),
                                ("end", 8, 113, None),
                            ),
                            self.get_expected_result(
                                336 + corrids[2], corrids[2], 113, "end"
                            ),
                            precreated_shm2_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}_{}".format(
                                self._testMethodName, corrids[2]
                            ),
                            "using_dynamic_batcher": True,
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )
                threads.append(
                    threading.Thread(
                        target=self.check_sequence_shape_tensor_io,
                        args=(
                            model_name,
                            dtype,
                            corrids[3],
                            (None, None),
                            (
                                ("start", 2, 1111, None),
                                (None, 4, 1112, None),
                                ("end", 8, 1113, None),
                            ),
                            self.get_expected_result(
                                3336 + corrids[3], corrids[3], 1113, "end"
                            ),
                            precreated_shm3_handles,
                        ),
                        kwargs={
                            "sequence_name": "{}_{}".format(
                                self._testMethodName, corrids[3]
                            ),
                            "using_dynamic_batcher": True,
                            "shape_tensor_input_dtype": shape_tensor_input_dtype,
                        },
                    )
                )

                for t in threads:
                    t.start()
                    if sleep_secs > 0:
                        time.sleep(sleep_secs)
                for t in threads:
                    t.join()
                self.check_deferred_exception()
                self.check_status(model_name, {4: 3}, 3, 12)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            finally:
                if TEST_SYSTEM_SHARED_MEMORY:
                    self.cleanup_shm_regions(precreated_shm0_handles)
                    self.cleanup_shm_regions(precreated_shm1_handles)
                    self.cleanup_shm_regions(precreated_shm2_handles)
                    self.cleanup_shm_regions(precreated_shm3_handles)

    def test_dynaseq_identical_shape_values_series(self):

        self._multi_sequence_identical_shape_impl(1)

    def test_dynaseq_identical_shape_values_parallel(self):

        self._multi_sequence_identical_shape_impl(0)

    def test_dynaseq_different_shape_values_series(self):

        self._multi_sequence_different_shape_impl(1)

    def test_dynaseq_different_shape_values_parallel(self):

        self._multi_sequence_different_shape_impl(0)


if __name__ == "__main__":
    unittest.main()
