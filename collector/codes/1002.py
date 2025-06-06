import sys

sys.path.append("../../common")

import os
import unittest

import pytest
import shm_util
import tritonclient.grpc as grpcclient
from tritonclient.utils import *


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


ALLOWED_FAILURE_EXIT_CODE = 123


class TestInferShmLeak:
    def _run_unittest(self, model_name):
        with grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001") as client:

            result = client.infer(model_name, [], client_timeout=240)
            output0 = result.as_numpy("OUTPUT0")

            assert output0 == [1], f"python_unittest failed for model {model_name}"

    def test_shm_leak(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()
        model_name = os.environ.get("MODEL_NAME", "default_model")

        try:
            with self._shm_leak_detector.Probe() as shm_probe:
                self._run_unittest(model_name)
        except AssertionError as e:
            if "Known shared memory leak of 480 bytes detected" in str(e):
                pytest.exit(str(e), returncode=ALLOWED_FAILURE_EXIT_CODE)
            else:
                raise e
