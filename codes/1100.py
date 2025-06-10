import sys

sys.path.append("../common")

import os
import random
import threading
import time
import unittest
from builtins import str
from functools import partial

import numpy as np
import sequence_util as su
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

TEST_SYSTEM_SHARED_MEMORY = bool(int(os.environ.get("TEST_SYSTEM_SHARED_MEMORY", 0)))
TEST_CUDA_SHARED_MEMORY = bool(int(os.environ.get("TEST_CUDA_SHARED_MEMORY", 0)))

USE_GRPC = os.environ.get("USE_GRPC", 1) != "0"
USE_HTTP = os.environ.get("USE_HTTP", 1) != "0"
assert USE_GRPC or USE_HTTP, "USE_GRPC or USE_HTTP must be non-zero"
if USE_GRPC and USE_HTTP:
    _protocols = ("http", "grpc")
elif USE_GRPC:
    _protocols = ("grpc",)
else:
    _protocols = ("http",)

BACKENDS = os.environ.get("BACKENDS", "onnx plan custom python")
ENSEMBLES = bool(int(os.environ.get("ENSEMBLES", 1)))

NO_BATCHING = int(os.environ["NO_BATCHING"]) == 1
MODEL_INSTANCES = int(os.environ["MODEL_INSTANCES"])
IMPLICIT_STATE = int(os.environ["IMPLICIT_STATE"]) == 1


INITIAL_STATE_FILE = int(os.environ["INITIAL_STATE_FILE"]) == 1

_trials = ()
if NO_BATCHING:
    for backend in BACKENDS.split(" "):
        if backend != "custom":
            _trials += (backend + "_nobatch",)
elif os.environ["BATCHER_TYPE"] == "VARIABLE":
    for backend in BACKENDS.split(" "):
        if (backend != "libtorch") and (backend != "custom"):
            _trials += (backend,)
else:
    _trials = BACKENDS.split(" ")


ENSEMBLE_PREFIXES = ["simple_", "sequence_", "fan_"]

if ENSEMBLES:
    res = []
    for trial in _trials:
        res.append(trial)
        if "custom" in trial:
            continue
        for ensemble_prefix in ENSEMBLE_PREFIXES:
            res.append(ensemble_prefix + trial)
    _trials = tuple(res)

_ragged_batch_supported_trials = list()
if "custom" in _trials:
    _ragged_batch_supported_trials = ("custom",)


_ragged_batch_not_supported_trials = list()
if os.environ["BATCHER_TYPE"] == "VARIABLE":
    if "custom" in _trials:
        _ragged_batch_not_supported_trials.append("custom")
    if "plan" in _trials:
        _ragged_batch_not_supported_trials.append("plan")
    if "onnx" in _trials:
        _ragged_batch_not_supported_trials.append("onnx")

_max_sequence_idle_ms = 5000


def is_ensemble(model_name):
    for prefix in ENSEMBLE_PREFIXES:
        if model_name.startswith(prefix):
            return True
    return False


class SequenceBatcherTest(su.SequenceBatcherTestUtil):
    def get_datatype(self, trial):

        if "plan" in trial:
            return (np.float32,)
        if "custom" in trial:
            return (np.int32,)

        if IMPLICIT_STATE:
            if "onnx" in trial:
                return (np.dtype(object), np.int32, np.bool_)
            if NO_BATCHING:
                if "libtorch" in trial:
                    return (np.dtype(object), np.int32, np.bool_)

        return (np.int32, np.bool_)

    def get_expected_result(self, expected_result, value, trial, flag_str=None):

        if (
            (not NO_BATCHING and ("custom" not in trial))
            or ("plan" in trial)
            or ("onnx" in trial)
        ) or ("libtorch" in trial):
            expected_result = value
            if (flag_str is not None) and ("start" in flag_str):
                expected_result += 1
        return expected_result

    def get_expected_result_implicit(
        self, expected_result, value, trial, flag_str=None, dtype=None
    ):
        if dtype == np.dtype(object) and trial.startswith("onnx"):
            return value

        if INITIAL_STATE_FILE:

            return expected_result + 100
        else:
            return expected_result

    def test_simple_sequence(self):

        for trial in _trials:

            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)

                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)

                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue

                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(45, 9, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                45, 9, trial, "end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            5,
                            (4000, None),
                            (
                                ("start", 1, None, None),
                                (None, 2, None, None),
                                (None, 3, None, None),
                                (None, 4, None, None),
                                (None, 5, None, None),
                                (None, 6, None, None),
                                (None, 7, None, None),
                                (None, 8, None, None),
                                ("end", 9, None, None),
                            ),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.check_status(
                            model_name, {1: 9 * (idx + 1)}, 9 * (idx + 1), 9 * (idx + 1)
                        )
                    except Exception as ex:
                        self.assertTrue(False, "unexpected error {}".format(ex))

    def test_length1_sequence(self):

        for trial in _trials:

            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)

                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)

                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue

                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(42, 42, trial, "start,end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                42, 42, trial, "start,end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            99,
                            (4000, None),
                            (("start,end", 42, None, None),),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.check_status(
                            model_name, {1: idx + 1}, (idx + 1), (idx + 1)
                        )
                    except Exception as ex:
                        self.assertTrue(False, "unexpected error {}".format(ex))

    def test_batch_size(self):

        if (MODEL_INSTANCES == 4) or NO_BATCHING:
            return

        for trial in _trials:

            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)

                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)

                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue

                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(10, 9, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                10, 9, trial, "end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            27,
                            (4000, None),
                            (("start", 1, None, None), ("end", 9, None, None)),
                            expected_result,
                            protocol,
                            batch_size=2,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.assertTrue(False, "expected error")
                    except Exception as ex:
                        for prefix in ENSEMBLE_PREFIXES:
                            if model_name.startswith(prefix):
                                base_model_name = model_name[(len(prefix)) :]
                                self.assertTrue(
                                    ex.message().startswith(
                                        str(
                                            "in ensemble '{}', "
                                            + "inference request to model '{}' must specify "
                                            + "batch-size 1 due to requirements of sequence "
                                            + "batcher"
                                        ).format(model_name, base_model_name)
                                    )
                                )
                                return
                        self.assertTrue(
                            ex.message().startswith(
                                str(
                                    "inference request to model '{}' must specify "
                                    + "batch-size 1 due to requirements of sequence "
                                    + "batcher"
                                ).format(model_name)
                            )
                        )

    def test_no_correlation_id(self):

        for trial in _trials:

            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)
                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)

                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue

                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(10, 9, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                10, 9, trial, "end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            0,
                            (4000, None),
                            (("start", 1, None, None), ("end", 9, None, None)),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.assertTrue(False, "expected error")
                    except Exception as ex:
                        for prefix in ENSEMBLE_PREFIXES:
                            if model_name.startswith(prefix):
                                base_model_name = model_name[(len(prefix)) :]
                                self.assertTrue(
                                    ex.message().startswith(
                                        str(
                                            "in ensemble '{}', "
                                            + "inference request to model '{}' must specify a "
                                            + "non-zero or non-empty correlation ID"
                                        ).format(model_name, base_model_name)
                                    )
                                )
                                return
                        self.assertTrue(
                            ex.message().startswith(
                                str(
                                    "inference request to model '{}' must specify a "
                                    + "non-zero or non-empty correlation ID"
                                ).format(model_name)
                            )
                        )

    def test_no_sequence_start(self):

        for trial in _trials:

            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)
                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)

                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue

                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )

                        expected_result = (
                            self.get_expected_result(6, 3, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                6, 3, trial, "end", dtype
                            )
                        )
                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            37469245,
                            (4000, None),
                            (
                                (None, 1, None, None),
                                (None, 2, None, None),
                                ("end", 3, None, None),
                            ),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.assertTrue(False, "expected error")
                    except Exception as ex:
                        print(model_name + "-> " + ex.message())
                        for prefix in ENSEMBLE_PREFIXES:
                            if model_name.startswith(prefix):
                                base_model_name = model_name[(len(prefix)) :]
                                self.assertTrue(
                                    ex.message().startswith(
                                        str(
                                            "in ensemble '{}', "
                                            + "inference request for sequence 37469245 to "
                                            + "model '{}' must specify the START flag on the first "
                                            + "request of the sequence"
                                        ).format(model_name, base_model_name)
                                    )
                                )
                                return
                        self.assertTrue(
                            ex.message().startswith(
                                str(
                                    "inference request for sequence 37469245 to "
                                    + "model '{}' must specify the START flag on the first "
                                    + "request of the sequence"
                                ).format(model_name)
                            )
                        )

    def test_no_sequence_start2(self):

        for trial in _trials:

            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)
                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)

                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue

                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(6, 3, trial, None)
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                6, 3, trial, None, dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            3,
                            (4000, None),
                            (
                                ("start", 1, None, None),
                                (None, 2, None, None),
                                ("end", 3, None, None),
                                (None, 55, None, None),
                            ),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_status(
                            model_name, {1: 3 * (idx + 1)}, 3 * (idx + 1), 3 * (idx + 1)
                        )
                        self.check_deferred_exception()
                        self.assertTrue(False, "expected error")
                    except Exception as ex:
                        for prefix in ENSEMBLE_PREFIXES:
                            if model_name.startswith(prefix):
                                base_model_name = model_name[(len(prefix)) :]
                                self.assertTrue(
                                    ex.message().startswith(
                                        str(
                                            "in ensemble '{}', "
                                            + "inference request for sequence 3 to model '{}' must "
                                            + "specify the START flag on the first request of "
                                            + "the sequence"
                                        ).format(model_name, base_model_name)
                                    )
                                )
                                return
                        self.assertTrue(
                            ex.message().startswith(
                                str(
                                    "inference request for sequence 3 to model '{}' must "
                                    + "specify the START flag on the first request of "
                                    + "the sequence"
                                ).format(model_name)
                            )
                        )

    def test_no_sequence_end(self):

        for trial in _trials:

            for idx, protocol in enumerate(_protocols):
                dtypes = self.get_datatype(trial)
                for dtype in dtypes:
                    model_name = tu.get_sequence_model_name(trial, dtype)

                    if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                        dtype == np.bool_
                    ):
                        continue

                    if dtype == np.bool_:
                        dtype = np.int32

                    self.clear_deferred_exceptions()
                    try:
                        self.check_setup(model_name)
                        self.assertNotIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                        self.assertNotIn(
                            "TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ
                        )
                        expected_result = (
                            self.get_expected_result(51, 9, trial, "end")
                            if not IMPLICIT_STATE
                            else self.get_expected_result_implicit(
                                51, 9, trial, "end", dtype
                            )
                        )

                        self.check_sequence(
                            trial,
                            model_name,
                            dtype,
                            4566,
                            (4000, None),
                            (
                                ("start", 1, None, None),
                                (None, 2, None, None),
                                ("start", 42, None, None),
                                ("end", 9, None, None),
                            ),
                            expected_result,
                            protocol,
                            sequence_name="{}_{}".format(
                                self._testMethodName, protocol
                            ),
                        )

                        self.check_deferred_exception()
                        self.check_status(
                            model_name, {1: 4 * (idx + 1)}, 4 * (idx + 1), 4 * (idx + 1)
                        )
                    except Exception as ex:
                        self.assertTrue(False, "unexpected error {}".format(ex))

    def test_half_batch(self):

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3, 4), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (0, 9, 5, 13), dtype, 1
                )

                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 8)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    expected_result = (
                        self.get_expected_result(10, 4, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            10, 4, trial, "end", dtype
                        )
                    )

                    threads = []
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                987,
                                (None, None),
                                (
                                    ("start", 1, None),
                                    (None, 2, None),
                                    (None, 3, None),
                                    ("end", 4, None),
                                ),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(27, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            27, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                988,
                                (None, None),
                                (
                                    ("start", 0, None),
                                    (None, 9, None),
                                    (None, 5, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 8}, 8, 8)
                    else:
                        stats_batch_size = 2 if MODEL_INSTANCES == 1 else 1
                        exec_cnt = 4 if MODEL_INSTANCES == 1 else 8
                        self.check_status(
                            model_name,
                            {stats_batch_size: 4 * min(2, MODEL_INSTANCES)},
                            exec_cnt,
                            8,
                        )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)

    def test_skip_batch(self):

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13, 14), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113, 1114), dtype, 3
                )

                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(4, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            4, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(50, 14, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            50, 14, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    (None, 13, None),
                                    ("end", 14, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(224, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            224, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (("start", 111, None), ("end", 113, None)),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(4450, 1114, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            4450, 1114, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    (None, 1113, None),
                                    ("end", 1114, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[1].start()
                    threads[3].start()
                    time.sleep(3)
                    threads[0].start()
                    threads[2].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:

                        if MODEL_INSTANCES == 1:
                            self.check_status(model_name, {2: 2, 4: 2}, 4, 12)
                        elif MODEL_INSTANCES == 2:
                            self.check_status(model_name, {2: 4, 1: 4}, 8, 12)
                        elif MODEL_INSTANCES == 4:
                            self.check_status(model_name, {1: 12}, 12, 12)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)

    def test_full_batch(self):

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )

                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads = []
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    expected_result = (
                        self.get_expected_result(36, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        self.check_status(
                            model_name,
                            {(4 / MODEL_INSTANCES): (3 * MODEL_INSTANCES)},
                            3 * MODEL_INSTANCES,
                            12,
                        )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)

    def test_ragged_batch(self):

        if MODEL_INSTANCES != 1:
            return

        for trial in _ragged_batch_not_supported_trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0, tensor_shape=(2,)
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1, tensor_shape=(2,)
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2, tensor_shape=(1,)
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3, tensor_shape=(3,)
                )

                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6 * 2, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (2,),
                            },
                        )
                    )

                    expected_result = (
                        self.get_expected_result(36 * 2, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (2,),
                            },
                        )
                    )
                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (1,),
                            },
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336 * 3, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (3,),
                            },
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    time.sleep(3)
                    threads[3].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        self.check_status(model_name, {4: 9}, 9, 12)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)

    def test_ragged_batch_allowed(self):

        if MODEL_INSTANCES != 1:
            return

        for trial in _ragged_batch_supported_trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0, tensor_shape=(2,)
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1, tensor_shape=(2,)
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2, tensor_shape=(1,)
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3, tensor_shape=(3,)
                )
                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []

                    expected_result = (
                        self.get_expected_result(6 * 2, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6 * 2, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (2,),
                            },
                        )
                    )

                    expected_result = (
                        self.get_expected_result(36 * 2, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36 * 2, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (2,),
                            },
                        )
                    )
                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (1,),
                            },
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336 * 3, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336 * 3, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={
                                "sequence_name": "{}".format(self._testMethodName),
                                "tensor_shape": (3,),
                            },
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        self.check_status(model_name, {4: 3}, 3, 12)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)

    def test_backlog(self):

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111, 11112, 11113), dtype, 4
                )

                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(36, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    expected_result = (
                        self.get_expected_result(33336, 11113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            33336, 11113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1005,
                                (None, None),
                                (
                                    ("start", 11111, None),
                                    (None, 11112, None),
                                    ("end", 11113, None),
                                ),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 15}, 15, 15)
                    else:
                        if MODEL_INSTANCES == 1:
                            self.check_status(model_name, {4: 3, 1: 3}, 6, 15)
                        elif MODEL_INSTANCES == 2:
                            self.check_status(model_name, {2: 6, 1: 3}, 9, 15)
                        else:
                            self.check_status(model_name, {1: 15}, 15, 15)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)

    def test_backlog_fill(self):

        if MODEL_INSTANCES != 1:
            return

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111,), dtype, 4
                )
                precreated_shm5_handles = self.precreate_register_regions(
                    (22222,), dtype, 5
                )

                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 10
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 2
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(24, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            24, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (("start", 11, None), ("end", 13, None)),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(224, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            224, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (("start", 111, None), ("end", 113, None)),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(11111, 11111, trial, "start,end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            11111, 11111, trial, "start,end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1005,
                                (None, None),
                                (("start,end", 11111, None),),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(22222, 22222, trial, "start,end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            22222, 22222, trial, "start,end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1006,
                                (None, None),
                                (("start,end", 22222, None),),
                                expected_result,
                                precreated_shm5_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    threads[3].start()
                    time.sleep(3)
                    threads[4].start()
                    threads[5].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 12}, 12, 12)
                    else:
                        self.check_status(model_name, {4: 3}, 3, 12)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)
                        self.cleanup_shm_regions(precreated_shm5_handles)

    def test_backlog_fill_no_end(self):

        if MODEL_INSTANCES != 1:
            return

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111,), dtype, 4
                )
                precreated_shm5_handles = self.precreate_register_regions(
                    (22222, 22223, 22224), dtype, 5
                )

                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 10
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 3
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(24, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            24, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (("start", 11, None), ("end", 13, None)),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(224, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            224, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (("start", 111, None), ("end", 113, None)),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(11111, 11111, trial, "start,end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            11111, 11111, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1005,
                                (None, None),
                                (("start,end", 11111, None),),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(66669, 22224, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            66669, 22224, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1006,
                                (None, None),
                                (
                                    ("start", 22222, None),
                                    (None, 22223, None),
                                    ("end", 22224, 2000),
                                ),
                                expected_result,
                                precreated_shm5_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    time.sleep(2)
                    threads[1].start()
                    time.sleep(2)
                    threads[2].start()
                    time.sleep(2)
                    threads[3].start()
                    time.sleep(2)
                    threads[4].start()
                    time.sleep(2)
                    threads[5].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 14}, 14, 14)
                    else:

                        self.check_status(model_name, {4: 3, 3: 2}, 5, 14)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)
                        self.cleanup_shm_regions(precreated_shm5_handles)

    def test_backlog_same_correlation_id(self):

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 2, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111, 11113), dtype, 4
                )

                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 12
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 2
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(6, 3, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            6, 3, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), (None, 2, None), ("end", 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(36, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            36, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(336, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            336, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(3336, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            3336, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(22224, 11113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            22224, 11113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (("start", 11111, None), ("end", 11113, None)),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    threads[3].start()
                    time.sleep(3)
                    threads[4].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 14}, 14, 14)
                    else:
                        if MODEL_INSTANCES != 4:
                            batch_exec = {
                                (4 / MODEL_INSTANCES): (3 * MODEL_INSTANCES),
                                1: 2,
                            }
                        else:
                            batch_exec = {1: (3 * MODEL_INSTANCES) + 2}
                        self.check_status(
                            model_name, batch_exec, (3 * MODEL_INSTANCES) + 2, 14
                        )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)

    def test_backlog_same_correlation_id_no_end(self):

        if MODEL_INSTANCES != 1:
            return

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111, 11113), dtype, 4
                )
                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 16
                    )
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(4, 3, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(4, 3, trial, None, dtype)
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 1, None), (None, 3, None)),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(48, 13, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            48, 13, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                    (None, 12, None),
                                    ("end", 13, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(448, 113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            448, 113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (
                                    ("start", 111, None),
                                    (None, 112, None),
                                    (None, 112, None),
                                    ("end", 113, None),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(4448, 1113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            4448, 1113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, None),
                                    (None, 1112, None),
                                    ("end", 1113, None),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(22224, 11113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            22224, 11113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (("start", 11111, None), ("end", 11113, None)),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    threads[3].start()
                    time.sleep(2)
                    threads[4].start()
                    for t in threads:
                        t.join()
                    self.check_deferred_exception()
                    if is_ensemble(model_name):

                        self.check_status(model_name, {1: 16}, 16, 16)
                    else:
                        self.check_status(model_name, {4: 4}, 4, 16)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)

    def test_backlog_sequence_timeout(self):

        if MODEL_INSTANCES != 1:
            return

        for trial in _trials:
            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if (any(word in trial for word in ENSEMBLE_PREFIXES)) and (
                    dtype == np.bool_
                ):
                    continue

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1, 3), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12, 12, 13), dtype, 1
                )
                precreated_shm2_handles = self.precreate_register_regions(
                    (111, 112, 112, 113), dtype, 2
                )
                precreated_shm3_handles = self.precreate_register_regions(
                    (1111, 1112, 1112, 1113), dtype, 3
                )
                precreated_shm4_handles = self.precreate_register_regions(
                    (11111, 11113), dtype, 4
                )
                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 4)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(4, 3, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(4, 3, trial, None, dtype)
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (None, None),
                                (
                                    ("start", 1, None),
                                    (None, 3, _max_sequence_idle_ms + 1000),
                                ),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(48, 13, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            48, 13, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (None, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, _max_sequence_idle_ms / 2),
                                    (None, 12, _max_sequence_idle_ms / 2),
                                    ("end", 13, _max_sequence_idle_ms / 2),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(448, 113, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            448, 113, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1003,
                                (None, None),
                                (
                                    ("start", 111, None),
                                    (None, 112, _max_sequence_idle_ms / 2),
                                    (None, 112, _max_sequence_idle_ms / 2),
                                    ("end", 113, _max_sequence_idle_ms / 2),
                                ),
                                expected_result,
                                precreated_shm2_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(4448, 1113, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            4448, 1113, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1004,
                                (None, None),
                                (
                                    ("start", 1111, None),
                                    (None, 1112, _max_sequence_idle_ms / 2),
                                    (None, 1112, _max_sequence_idle_ms / 2),
                                    ("end", 1113, _max_sequence_idle_ms / 2),
                                ),
                                expected_result,
                                precreated_shm3_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(22224, 11113, trial, "end")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            22224, 11113, trial, "end", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1005,
                                (None, None),
                                (("start", 11111, None), ("end", 11113, None)),
                                expected_result,
                                precreated_shm4_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    threads[1].start()
                    threads[2].start()
                    threads[3].start()
                    time.sleep(2)
                    threads[4].start()
                    for t in threads:
                        t.join()

                    self.check_deferred_exception()
                    self.assertTrue(False, "expected error")
                except Exception as ex:
                    for prefix in ENSEMBLE_PREFIXES:
                        if model_name.startswith(prefix):
                            base_model_name = model_name[(len(prefix)) :]
                            self.assertTrue(
                                ex.message().startswith(
                                    str(
                                        "in ensemble '{}', "
                                        + "inference request for sequence 1001 to "
                                        + "model '{}' must specify the START flag on the first "
                                        + "request of the sequence"
                                    ).format(model_name, base_model_name)
                                )
                            )
                            return
                    self.assertTrue(
                        ex.message().startswith(
                            str(
                                "inference request for sequence 1001 to "
                                + "model '{}' must specify the START flag on the first "
                                + "request of the sequence"
                            ).format(model_name)
                        )
                    )
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)
                        self.cleanup_shm_regions(precreated_shm2_handles)
                        self.cleanup_shm_regions(precreated_shm3_handles)
                        self.cleanup_shm_regions(precreated_shm4_handles)

    def test_queue_delay_no_min_util(self):

        for trial in _trials:
            is_ensemble = False
            for prefix in ENSEMBLE_PREFIXES:
                if prefix in trial:
                    is_ensemble = True
                    break
            if is_ensemble:
                continue

            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype)

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1,), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12), dtype, 1
                )
                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 2)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(1, 1, trial, "start")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            1, 1, trial, "start", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (2000, None),
                                (("start", 1, None),),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(23, 12, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            23, 12, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (2000, None),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    time.sleep(1)
                    threads[1].start()
                    for t in threads:
                        t.join()

                    self.check_deferred_exception()
                    self.check_status(model_name, {2: 2}, 2, 3)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)

    def test_queue_delay_half_min_util(self):

        for trial in _trials:
            is_ensemble = False
            for prefix in ENSEMBLE_PREFIXES:
                if prefix in trial:
                    is_ensemble = True
                    break
            if is_ensemble:
                continue

            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype) + "_half"

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1,), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12), dtype, 1
                )
                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 2)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(1, 1, trial, "start")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            1, 1, trial, "start", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (2000, None),
                                (("start", 1, None),),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(23, 12, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            23, 12, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (4000, 3000),
                                (
                                    ("start", 11, None),
                                    (None, 12, None),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    time.sleep(1)
                    threads[1].start()
                    for t in threads:
                        t.join()

                    self.check_deferred_exception()
                    self.check_status(model_name, {2: 2}, 2, 3)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)

    def test_queue_delay_full_min_util(self):

        for trial in _trials:
            is_ensemble = False
            for prefix in ENSEMBLE_PREFIXES:
                if prefix in trial:
                    is_ensemble = True
                    break
            if is_ensemble:
                continue

            dtypes = self.get_datatype(trial)
            for dtype in dtypes:
                model_name = tu.get_sequence_model_name(trial, dtype) + "_full"

                if dtype == np.bool_:
                    dtype = np.int32

                self.clear_deferred_exceptions()

                precreated_shm0_handles = self.precreate_register_regions(
                    (1,), dtype, 0
                )
                precreated_shm1_handles = self.precreate_register_regions(
                    (11, 12), dtype, 1
                )
                try:
                    self.check_setup(model_name)

                    self.assertIn("TRITONSERVER_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(int(os.environ["TRITONSERVER_DELAY_SCHEDULER"]), 2)
                    self.assertIn("TRITONSERVER_BACKLOG_DELAY_SCHEDULER", os.environ)
                    self.assertEqual(
                        int(os.environ["TRITONSERVER_BACKLOG_DELAY_SCHEDULER"]), 0
                    )

                    threads = []
                    expected_result = (
                        self.get_expected_result(1, 1, trial, "start")
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            1, 1, trial, "start", dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1001,
                                (4000, 3000),
                                (("start", 1, None),),
                                expected_result,
                                precreated_shm0_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )
                    expected_result = (
                        self.get_expected_result(23, 12, trial, None)
                        if not IMPLICIT_STATE
                        else self.get_expected_result_implicit(
                            23, 12, trial, None, dtype
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=self.check_sequence_async,
                            args=(
                                trial,
                                model_name,
                                dtype,
                                1002,
                                (6000, 5000),
                                (
                                    ("start", 11, None),
                                    (None, 12, 2000),
                                ),
                                expected_result,
                                precreated_shm1_handles,
                            ),
                            kwargs={"sequence_name": "{}".format(self._testMethodName)},
                        )
                    )

                    threads[0].start()
                    time.sleep(1)
                    threads[1].start()
                    for t in threads:
                        t.join()

                    self.check_deferred_exception()
                    self.check_status(model_name, {2: 2}, 2, 3)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                finally:
                    if TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY:
                        self.cleanup_shm_regions(precreated_shm0_handles)
                        self.cleanup_shm_regions(precreated_shm1_handles)


class SequenceBatcherRequestTimeoutTest(su.SequenceBatcherTestUtil):
    def setUp(self):
        super(SequenceBatcherRequestTimeoutTest, self).setUp()

        self.server_address_ = (
            os.environ.get("TRITONSERVER_IPADDR", "localhost") + ":8001"
        )

        self.model_name_ = "custom_sequence_int32_timeout"
        self.tensor_data_ = np.ones(shape=[1, 1], dtype=np.int32)
        self.inputs_ = [grpcclient.InferInput("INPUT0", [1, 1], "INT32")]
        self.inputs_[0].set_data_from_numpy(self.tensor_data_)
        self.expected_out_seq_ = [
            ("OUTPUT0", self.tensor_data_),
            ("OUTPUT0", self.tensor_data_),
            ("OUTPUT0", self.tensor_data_),
        ]

    def send_sequence_with_timeout(
        self, seq_id, callback, timeout_us=3000000, request_pause_sec=0
    ):
        with grpcclient.InferenceServerClient(self.server_address_) as triton_client:
            triton_client.start_stream(callback=callback)
            triton_client.async_stream_infer(
                self.model_name_,
                self.inputs_,
                sequence_id=seq_id,
                sequence_start=True,
                timeout=timeout_us,
            )
            if request_pause_sec != 0:
                time.sleep(request_pause_sec)
            triton_client.async_stream_infer(
                self.model_name_, self.inputs_, sequence_id=seq_id, timeout=timeout_us
            )
            if request_pause_sec != 0:
                time.sleep(request_pause_sec)
            triton_client.async_stream_infer(
                self.model_name_,
                self.inputs_,
                sequence_id=seq_id,
                sequence_end=True,
                timeout=timeout_us,
            )

    def test_request_timeout(self):

        seq1_res = []
        seq2_res = []
        seq1_callback = lambda result, error: seq1_res.append((result, error))
        seq2_callback = lambda result, error: seq2_res.append((result, error))

        threads = []
        threads.append(
            threading.Thread(
                target=self.send_sequence_with_timeout, args=(1, seq1_callback)
            )
        )
        threads.append(
            threading.Thread(
                target=self.send_sequence_with_timeout, args=(2, seq2_callback)
            )
        )
        threads[0].start()
        time.sleep(1)
        threads[1].start()
        for t in threads:
            t.join()

        for idx in range(len(seq1_res)):
            result, error = seq1_res[idx]
            self.assertIsNone(
                error,
                "Expect successful inference for sequence 1 requests, got error: {}".format(
                    error
                ),
            )
            out = result.as_numpy(self.expected_out_seq_[idx][0])
            expected_out = self.expected_out_seq_[idx][1]
            np.testing.assert_allclose(
                out,
                expected_out,
                err_msg="Unexpected output tensor: expect {}, got {}".format(
                    expected_out, out
                ),
            )

        for _, error in seq2_res:
            self.assertIsNotNone(error, "Expect error for sequence 2 requests")
            with self.assertRaisesRegex(
                InferenceServerException,
                "timeout of the corresponding sequence has been expired",
                msg="Unexpected error: {}".format(error),
            ):
                raise error

    def test_send_request_after_timeout(self):

        seq1_res = []
        seq2_res = []
        seq1_callback = lambda result, error: seq1_res.append((result, error))
        seq2_callback = lambda result, error: seq2_res.append((result, error))

        threads = []
        threads.append(
            threading.Thread(
                target=self.send_sequence_with_timeout, args=(1, seq1_callback)
            )
        )

        threads.append(
            threading.Thread(
                target=self.send_sequence_with_timeout,
                args=(2, seq2_callback),
                kwargs={"request_pause_sec": 2},
            )
        )
        threads[0].start()
        time.sleep(1)
        threads[1].start()
        for t in threads:
            t.join()

        for _, error in seq2_res[0:-1]:
            self.assertIsNotNone(error, "Expect error for sequence 2 requests")
            with self.assertRaisesRegex(
                InferenceServerException,
                "timeout of the corresponding sequence has been expired",
                msg="Unexpected error: {}".format(error),
            ):
                raise error
        _, last_err = seq2_res[-1]
        self.assertIsNotNone(last_err, "Expect error for sequence 2 requests")
        with self.assertRaisesRegex(
            InferenceServerException,
            "must specify the START flag on the first request",
            msg="Unexpected error: {}".format(last_err),
        ):
            raise last_err


class SequenceBatcherPreserveOrderingTest(su.SequenceBatcherTestUtil):
    def setUp(self):
        super().setUp()

        self.server_address_ = (
            os.environ.get("TRITONSERVER_IPADDR", "localhost") + ":8001"
        )

        self.model_name_ = "sequence_py"
        self.tensor_data_ = np.ones(shape=[1, 1], dtype=np.int32)
        self.inputs_ = [grpcclient.InferInput("INPUT0", [1, 1], "INT32")]
        self.inputs_[0].set_data_from_numpy(self.tensor_data_)
        self.triton_client = grpcclient.InferenceServerClient(self.server_address_)

        self.request_id_lock = threading.Lock()
        self.request_id = 1

    def send_sequence(self, seq_id, seq_id_map, req_id_map):
        if seq_id not in seq_id_map:
            seq_id_map[seq_id] = []

        start, middle, end = (True, False), (False, False), (False, True)

        seq_flags = [start, middle, end]
        for start_flag, end_flag in seq_flags:

            time.sleep(random.uniform(0.0, 1.0))

            with self.request_id_lock:
                req_id = self.request_id
                self.request_id += 1

                req_id_map[req_id] = seq_id
                seq_id_map[seq_id].append(req_id)

                self.triton_client.async_stream_infer(
                    self.model_name_,
                    self.inputs_,
                    sequence_id=seq_id,
                    sequence_start=start_flag,
                    sequence_end=end_flag,
                    timeout=None,
                    request_id=str(req_id),
                )

    def _test_sequence_ordering(self, preserve_ordering, decoupled):

        class SequenceResult:
            def __init__(self, seq_id, result, request_id):
                self.seq_id = seq_id
                self.result = result
                self.request_id = int(request_id)

        def full_callback(sequence_dict, sequence_list, result, error):

            if error:
                self.assertTrue(False, error)

            request_id = int(result.get_response().id)
            sequence_id = request_id_map[request_id]

            sequence_list.append(SequenceResult(sequence_id, result, request_id))

            sequence_dict[sequence_id].append(result)

        sequence_list = []

        sequence_dict = {}

        sequence_id_map = {}
        request_id_map = {}

        seq_callback = partial(full_callback, sequence_dict, sequence_list)
        self.triton_client.start_stream(callback=seq_callback)

        threads = []
        num_sequences = 10
        for i in range(num_sequences):

            sequence_id = i + 1

            sequence_dict[sequence_id] = []
            threads.append(
                threading.Thread(
                    target=self.send_sequence,
                    args=(sequence_id, sequence_id_map, request_id_map),
                )
            )

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        self.triton_client.stop_stream()

        self.assertGreater(len(sequence_dict), 0)
        self.assertGreater(len(sequence_list), 0)

        print(f"=== {preserve_ordering=} {decoupled=} ===")
        print("Outputs per Sequence:")
        for seq_id, sequence in sequence_dict.items():
            seq_outputs = [
                result.as_numpy("OUTPUT0").flatten().tolist() for result in sequence
            ]
            print(f"{seq_id}: {seq_outputs}")
            self.assertEqual(seq_outputs, sorted(seq_outputs))

        print("Request IDs per Sequence:")
        for seq_id in sequence_id_map:
            per_seq_request_ids = sequence_id_map[seq_id]
            print(f"{seq_id}: {per_seq_request_ids}")
            self.assertEqual(per_seq_request_ids, sorted(per_seq_request_ids))

        if preserve_ordering:
            request_ids = [s.request_id for s in sequence_list]
            print(f"Request IDs overall:\n{request_ids}")
            sequence_ids = [s.seq_id for s in sequence_list]
            print(f"Sequence IDs overall:\n{sequence_ids}")
            self.assertEqual(request_ids, sorted(request_ids))

        stats = self.triton_client.get_inference_statistics(
            model_name=self.model_name_, headers={}, as_json=True
        )
        model_stats = stats["model_stats"][0]
        self.assertEqual(model_stats["name"], self.model_name_)
        self.assertLess(
            int(model_stats["execution_count"]), int(model_stats["inference_count"])
        )

    def test_sequence_with_preserve_ordering(self):
        self.model_name_ = "seqpy_preserve_ordering_nondecoupled"
        self._test_sequence_ordering(preserve_ordering=True, decoupled=False)

    def test_sequence_without_preserve_ordering(self):
        self.model_name_ = "seqpy_no_preserve_ordering_nondecoupled"
        self._test_sequence_ordering(preserve_ordering=False, decoupled=False)


if __name__ == "__main__":
    unittest.main()
