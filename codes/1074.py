import concurrent.futures
import json
import os
import random
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from models.model_init_del.util import (
    disable_batching,
    enable_batching,
    get_count,
    reset_count,
    set_delay,
    update_instance_group,
    update_model_file,
    update_sequence_batching,
)
from tritonclient.utils import InferenceServerException


class TestInstanceUpdate(unittest.TestCase):
    _model_name = "model_init_del"

    def setUp(self):

        reset_count("initialize")
        reset_count("finalize")

        disable_batching()

        set_delay("initialize", 0)
        set_delay("infer", 0)

        update_sequence_batching("")

        self._triton = grpcclient.InferenceServerClient("localhost:8001")

    def tearDown(self):

        r = self._outcome.result
        passed = all(self != test_case for test_case, _ in r.errors + r.failures)
        if passed:

            return

        self._triton.unload_model(self._model_name)
        time.sleep(30)

    def _get_inputs(self, batching=False):
        self.assertIsInstance(batching, bool)
        if batching:
            shape = [random.randint(1, 2), random.randint(1, 16)]
        else:
            shape = [random.randint(1, 16)]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        return inputs

    def _infer(self, batching=False):
        self._triton.infer(self._model_name, self._get_inputs(batching))

    def _concurrent_infer(self, concurrency=4, batching=False):
        pool = concurrent.futures.ThreadPoolExecutor()
        stop = [False]

        def repeat_infer():
            while not stop[0]:
                self._infer(batching)

        infer_threads = [pool.submit(repeat_infer) for i in range(concurrency)]

        def stop_infer():
            stop[0] = True
            [t.result() for t in infer_threads]
            pool.shutdown()

        return stop_infer

    def _check_count(self, kind, expected_count, poll=False):
        self.assertIsInstance(poll, bool)
        if poll:
            timeout = 30
            poll_interval = 0.1
            max_retry = timeout / poll_interval
            num_retry = 0
            while num_retry < max_retry and get_count(kind) < expected_count:
                time.sleep(poll_interval)
                num_retry += 1
        self.assertEqual(get_count(kind), expected_count)

    def _load_model(self, instance_count, instance_config="", batching=False):

        enable_batching() if batching else disable_batching()

        self._update_instance_count(
            instance_count, 0, instance_config, batching=batching
        )

    def _update_instance_count(
        self,
        add_count,
        del_count,
        instance_config="",
        wait_for_finalize=False,
        batching=False,
    ):
        self.assertIsInstance(add_count, int)
        self.assertGreaterEqual(add_count, 0)
        self.assertIsInstance(del_count, int)
        self.assertGreaterEqual(del_count, 0)
        self.assertIsInstance(instance_config, str)
        prev_initialize_count = get_count("initialize")
        prev_finalize_count = get_count("finalize")
        new_initialize_count = prev_initialize_count + add_count
        new_finalize_count = prev_finalize_count + del_count
        if len(instance_config) == 0:
            prev_count = prev_initialize_count - prev_finalize_count
            new_count = prev_count + add_count - del_count
            instance_config = "{\ncount: " + str(new_count) + "\nkind: KIND_CPU\n}"
        update_instance_group(instance_config)
        self._triton.load_model(self._model_name)
        self._check_count("initialize", new_initialize_count)
        self._check_count("finalize", new_finalize_count, wait_for_finalize)
        self._infer(batching)

    def _unload_model(self, batching=False):
        prev_initialize_count = get_count("initialize")
        self._triton.unload_model(self._model_name)
        self._check_count("initialize", prev_initialize_count)
        self._check_count("finalize", prev_initialize_count, True)
        with self.assertRaises(InferenceServerException):
            self._infer(batching)

    def test_add_rm_add_instance_no_batching(self):
        self._load_model(3, batching=False)
        stop = self._concurrent_infer(batching=False)
        self._update_instance_count(1, 0, batching=False)
        self._update_instance_count(0, 1, batching=False)
        self._update_instance_count(1, 0, batching=False)
        stop()
        self._unload_model(batching=False)

    def test_add_rm_add_instance_with_batching(self):
        self._load_model(4, batching=True)
        stop = self._concurrent_infer(batching=True)
        self._update_instance_count(1, 0, batching=True)
        self._update_instance_count(0, 1, batching=True)
        self._update_instance_count(1, 0, batching=True)
        stop()
        self._unload_model(batching=True)

    def test_rm_add_rm_instance_no_batching(self):
        self._load_model(2, batching=False)
        stop = self._concurrent_infer(batching=False)
        self._update_instance_count(0, 1, batching=False)
        self._update_instance_count(1, 0, batching=False)
        self._update_instance_count(0, 1, batching=False)
        stop()
        self._unload_model(batching=False)

    def test_rm_add_rm_instance_with_batching(self):
        self._load_model(3, batching=True)
        stop = self._concurrent_infer(batching=True)
        self._update_instance_count(0, 1, batching=True)
        self._update_instance_count(1, 0, batching=True)
        self._update_instance_count(0, 1, batching=True)
        stop()
        self._unload_model(batching=True)

    def test_rm_instance_to_zero(self):
        self._load_model(1)

        self._update_instance_count(0, 0, "{\ncount: 0\nkind: KIND_CPU\n}")
        self._unload_model()

    def test_cpu_instance_update(self):
        self._load_model(8)
        self._update_instance_count(0, 4)
        self._update_instance_count(0, 3)
        self._update_instance_count(0, 0)
        time.sleep(0.1)
        self._update_instance_count(2, 0)
        self._update_instance_count(5, 0)
        self._unload_model()

    def test_gpu_instance_update(self):
        self._load_model(6, "{\ncount: 6\nkind: KIND_GPU\n}")
        self._update_instance_count(0, 2, "{\ncount: 4\nkind: KIND_GPU\n}")
        self._update_instance_count(3, 0, "{\ncount: 7\nkind: KIND_GPU\n}")
        self._unload_model()

    def test_gpu_cpu_instance_update(self):

        self._load_model(
            3, "{\ncount: 2\nkind: KIND_CPU\n},\n{\ncount: 1\nkind: KIND_GPU\n}"
        )

        self._update_instance_count(
            2, 1, "{\ncount: 1\nkind: KIND_CPU\n},\n{\ncount: 3\nkind: KIND_GPU\n}"
        )

        self._update_instance_count(
            0, 0, "{\ncount: 3\nkind: KIND_GPU\n},\n{\ncount: 1\nkind: KIND_CPU\n}"
        )
        time.sleep(0.1)

        self._update_instance_count(
            1, 1, "{\ncount: 2\nkind: KIND_GPU\n},\n{\ncount: 2\nkind: KIND_CPU\n}"
        )

        self._unload_model()

    def test_instance_name_update(self):

        self._load_model(
            3,
            '{\nname: "old_1"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "old_2"\ncount: 2\nkind: KIND_GPU\n}',
        )

        self._update_instance_count(
            0,
            0,
            '{\nname: "new_1"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "new_2"\ncount: 2\nkind: KIND_GPU\n}',
        )

        self._unload_model()

    def test_instance_signature(self):

        self._load_model(
            5,
            '{\nname: "GPU_group"\ncount: 2\nkind: KIND_GPU\n},\n{\nname: "CPU_group"\ncount: 3\nkind: KIND_CPU\n}',
        )

        self._update_instance_count(
            0,
            0,
            '{\nname: "CPU_1"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "CPU_2_3"\ncount: 2\nkind: KIND_CPU\n},\n{\nname: "GPU_1"\ncount: 1\nkind: KIND_GPU\n},\n{\nname: "GPU_2"\ncount: 1\nkind: KIND_GPU\n}',
        )
        time.sleep(0.1)

        self._update_instance_count(
            0,
            0,
            '{\nname: "CPU_group"\ncount: 3\nkind: KIND_CPU\n},\n{\nname: "GPU_group"\ncount: 2\nkind: KIND_GPU\n}',
        )
        time.sleep(0.1)

        self._update_instance_count(
            0,
            0,
            '{\nname: "GPU_1"\ncount: 1\nkind: KIND_GPU\n},\n{\nname: "GPU_2"\ncount: 1\nkind: KIND_GPU\n},\n{\nname: "CPU_1"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "CPU_2"\ncount: 1\nkind: KIND_CPU\n},\n{\nname: "CPU_3"\ncount: 1\nkind: KIND_CPU\n}',
        )

        self._unload_model()

    def test_invalid_config(self):

        self._load_model(8)

        update_instance_group("--- invalid config ---")
        with self.assertRaises(InferenceServerException):
            self._triton.load_model("model_init_del")

        self._update_instance_count(0, 4)

        self._unload_model()

    def test_model_file_update(self):
        self._load_model(5)
        update_model_file()
        self._update_instance_count(
            6, 5, "{\ncount: 6\nkind: KIND_CPU\n}", wait_for_finalize=True
        )
        self._unload_model()

    def test_non_instance_config_update(self):
        self._load_model(4, batching=False)
        enable_batching()
        self._update_instance_count(
            2,
            4,
            "{\ncount: 2\nkind: KIND_CPU\n}",
            wait_for_finalize=True,
            batching=True,
        )
        self._unload_model(batching=True)

    def test_load_api_with_config(self):

        self._load_model(1)

        config = self._triton.get_model_config(self._model_name, as_json=True)
        self.assertIn("config", config)
        self.assertIsInstance(config["config"], dict)
        config = config["config"]
        self.assertIn("instance_group", config)
        self.assertIsInstance(config["instance_group"], list)
        self.assertEqual(len(config["instance_group"]), 1)
        self.assertIn("count", config["instance_group"][0])
        self.assertIsInstance(config["instance_group"][0]["count"], int)

        config["instance_group"][0]["count"] += 1
        self.assertEqual(config["instance_group"][0]["count"], 2)

        self._triton.load_model(self._model_name, config=json.dumps(config))
        self._check_count("initialize", 2)
        self._check_count("finalize", 0)
        self._infer()

        self._unload_model()

    def test_update_while_inferencing(self):

        self._load_model(1)

        set_delay("infer", 10)
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            infer_start_time = time.time()
            infer_thread = pool.submit(self._infer)
            time.sleep(2)
            update_start_time = time.time()
            update_thread = pool.submit(self._triton.load_model, self._model_name)
            update_thread.result()
            update_end_time = time.time()
            infer_thread.result()
            infer_end_time = time.time()
        infer_time = infer_end_time - infer_start_time
        update_time = update_end_time - update_start_time

        self.assertGreaterEqual(infer_time, 10.0, "Invalid infer time")
        self.assertLess(update_time, 5.0, "Update blocked by infer")
        self._check_count("initialize", 2)
        self._check_count("finalize", 0)
        self._infer()

        self._unload_model()

    def test_infer_while_updating(self):

        self._load_model(1)

        set_delay("initialize", 10)
        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            update_start_time = time.time()
            update_thread = pool.submit(self._triton.load_model, self._model_name)
            time.sleep(2)
            infer_start_time = time.time()
            infer_thread = pool.submit(self._infer)
            infer_thread.result()
            infer_end_time = time.time()
            update_thread.result()
            update_end_time = time.time()
        update_time = update_end_time - update_start_time
        infer_time = infer_end_time - infer_start_time

        self.assertGreaterEqual(update_time, 10.0, "Invalid update time")
        self.assertLess(infer_time, 5.0, "Infer blocked by update")
        self._check_count("initialize", 2)
        self._check_count("finalize", 0)
        self._infer()

        self._unload_model()

    @unittest.skipUnless(
        "execution_count" in os.environ["RATE_LIMIT_MODE"],
        "Rate limiter precondition not met for this test",
    )
    def test_instance_resource_increase(self):

        self._load_model(
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 2\n}\n]\n}\n}',
        )

        self._update_instance_count(
            1,
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 8\n}\n]\n}\n}',
        )

        infer_count = 8
        infer_complete = [False for i in range(infer_count)]

        def infer():
            for i in range(infer_count):
                self._infer()
                infer_complete[i] = True

        with concurrent.futures.ThreadPoolExecutor() as pool:
            infer_thread = pool.submit(infer)
            time.sleep(infer_count / 2)
            self.assertNotIn(False, infer_complete, "Infer possibly stuck")
            infer_thread.result()

        self._unload_model()

    @unittest.skipUnless(
        os.environ["RATE_LIMIT_MODE"] == "execution_count_with_explicit_resource",
        "Rate limiter precondition not met for this test",
    )
    def test_instance_resource_increase_above_explicit(self):

        self._load_model(
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 2\n}\n]\n}\n}',
        )

        with self.assertRaises(InferenceServerException):
            self._update_instance_count(
                0,
                0,
                '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 32\n}\n]\n}\n}',
            )

        self._update_instance_count(
            1,
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 10\n}\n]\n}\n}',
        )

        self._unload_model()

    @unittest.skipUnless(
        "execution_count" in os.environ["RATE_LIMIT_MODE"],
        "Rate limiter precondition not met for this test",
    )
    def test_instance_resource_decrease(self):

        self._load_model(
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 4\n}\n]\n}\n}',
        )

        self._update_instance_count(
            1,
            1,
            '{\ncount: 1\nkind: KIND_CPU\nrate_limiter {\nresources [\n{\nname: "R1"\ncount: 3\n}\n]\n}\n}',
        )

        self._unload_model()

        time.sleep(1)
        log_path = os.path.join(
            os.environ["MODEL_LOG_DIR"],
            "instance_update_test.rate_limit_"
            + os.environ["RATE_LIMIT_MODE"]
            + ".server.log",
        )
        with open(log_path, mode="r", encoding="utf-8", errors="strict") as f:
            if os.environ["RATE_LIMIT_MODE"] == "execution_count":

                self.assertIn("Resource: R1\\t Count: 3", f.read())
            else:

                self.assertNotIn("Resource: R1\\t Count: 3", f.read())

    _direct_sequence_batching_str = (
        "direct { }\nmax_sequence_idle_microseconds: 8000000"
    )
    _oldest_sequence_batching_str = (
        "oldest { max_candidate_sequences: 4 }\nmax_sequence_idle_microseconds: 8000000"
    )

    def test_direct_scheduler_update_no_ongoing_sequences(self):
        self._test_scheduler_update_no_ongoing_sequences(
            self._direct_sequence_batching_str
        )

    def test_direct_scheduler_update_with_ongoing_sequences(self):
        self._test_scheduler_update_with_ongoing_sequences(
            self._direct_sequence_batching_str
        )

    def test_oldest_scheduler_update_no_ongoing_sequences(self):
        self._test_scheduler_update_no_ongoing_sequences(
            self._oldest_sequence_batching_str
        )

    def test_oldest_scheduler_update_with_ongoing_sequences(self):
        self._test_scheduler_update_with_ongoing_sequences(
            self._oldest_sequence_batching_str
        )

    def _test_scheduler_update_no_ongoing_sequences(self, sequence_batching_str):

        update_instance_group("{\ncount: 2\nkind: KIND_CPU\n}")
        update_sequence_batching(sequence_batching_str)
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 2)
        self._check_count("finalize", 0)

        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_start=True
        )
        self._triton.infer(self._model_name, self._get_inputs(), sequence_id=1)
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_end=True
        )

        update_instance_group("{\ncount: 4\nkind: KIND_CPU\n}")
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 4)
        self._check_count("finalize", 0)

        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_start=True
        )
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_end=True
        )

        update_instance_group("{\ncount: 3\nkind: KIND_CPU\n}")
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 4)
        self._check_count("finalize", 1, poll=True)

        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_start=True
        )
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_end=True
        )

        self._triton.unload_model(self._model_name)
        self._check_count("initialize", 4)
        self._check_count("finalize", 4, poll=True)

    def _test_scheduler_update_with_ongoing_sequences(self, sequence_batching_str):

        update_instance_group("{\ncount: 3\nkind: KIND_CPU\n}")
        update_sequence_batching(sequence_batching_str)
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 3)
        self._check_count("finalize", 0)

        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_start=True
        )
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=2, sequence_start=True
        )

        update_instance_group("{\ncount: 1\nkind: KIND_GPU\n}")
        self._triton.load_model(self._model_name)
        self._check_count("initialize", 4)
        self._check_count("finalize", 1, poll=True)

        self._triton.infer(self._model_name, self._get_inputs(), sequence_id=1)
        self._triton.infer(self._model_name, self._get_inputs(), sequence_id=2)
        self._check_count("finalize", 1)

        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=3, sequence_start=True
        )
        self._check_count("finalize", 1)

        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=1, sequence_end=True
        )
        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=2, sequence_end=True
        )
        self._check_count("finalize", 3, poll=True)

        self._triton.infer(
            self._model_name, self._get_inputs(), sequence_id=3, sequence_end=True
        )

        self._triton.unload_model(self._model_name)
        self._check_count("initialize", 4)
        self._check_count("finalize", 4, poll=True)


if __name__ == "__main__":
    unittest.main()
