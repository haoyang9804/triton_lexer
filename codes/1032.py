import time
import unittest
from functools import partial

import nvidia_smi
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class UnifiedClientProxy:
    def __init__(self, client):
        self.client_ = client

    def __getattr__(self, attr):
        forward_attr = getattr(self.client_, attr)
        if type(self.client_) == grpcclient.InferenceServerClient:
            if attr == "get_model_config":
                return lambda *args, **kwargs: forward_attr(
                    *args, **kwargs, as_json=True
                )["config"]
            elif attr == "get_inference_statistics":
                return partial(forward_attr, as_json=True)
        return forward_attr


class MemoryUsageTest(unittest.TestCase):
    def setUp(self):
        nvidia_smi.nvmlInit()
        self.gpu_handle_ = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        self.http_client_ = httpclient.InferenceServerClient(url="localhost:8000")
        self.grpc_client_ = grpcclient.InferenceServerClient(url="localhost:8001")

    def tearDown(self):
        nvidia_smi.nvmlShutdown()

    def report_used_gpu_memory(self):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handle_)
        return info.used

    def is_testing_backend(self, model_name, backend_name):
        return self.client_.get_model_config(model_name)["backend"] == backend_name

    def verify_recorded_usage(self, model_stat):
        recorded_gpu_usage = 0
        for usage in model_stat["memory_usage"]:
            if usage["type"] == "GPU":
                recorded_gpu_usage += int(usage["byte_size"])

        before_total_usage = self.report_used_gpu_memory()
        self.client_.unload_model(model_stat["name"])

        time.sleep(2)
        usage_delta = before_total_usage - self.report_used_gpu_memory()

        self.assertTrue(
            usage_delta * 0.9 <= recorded_gpu_usage <= usage_delta * 1.1,
            msg="For model {}, expect recorded usage to be in range [{}, {}], got {}".format(
                model_stat["name"],
                usage_delta * 0.9,
                usage_delta * 1.1,
                recorded_gpu_usage,
            ),
        )

    def test_onnx_http(self):
        self.client_ = UnifiedClientProxy(self.http_client_)
        model_stats = self.client_.get_inference_statistics()["model_stats"]
        for model_stat in model_stats:
            if self.is_testing_backend(model_stat["name"], "onnxruntime"):
                self.verify_recorded_usage(model_stat)

    def test_plan_grpc(self):
        self.client_ = UnifiedClientProxy(self.grpc_client_)
        model_stats = self.client_.get_inference_statistics()["model_stats"]
        for model_stat in model_stats:
            if self.is_testing_backend(model_stat["name"], "tensorrt"):
                self.verify_recorded_usage(model_stat)


if __name__ == "__main__":
    unittest.main()
