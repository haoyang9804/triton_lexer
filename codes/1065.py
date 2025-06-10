import os
import re
import sys
import unittest
from functools import partial

import numpy as np
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

sys.path.append("../common")
import test_util as tu

MILLIS_PER_SEC = 1000
FIRST_RESPONSE_HISTOGRAM = "nv_inference_first_response_histogram_ms"


def get_histogram_metric_key(
    metric_family, model_name, model_version, metric_type, le=""
):
    if metric_type in ["count", "sum"]:
        return f'{metric_family}_{metric_type}{{model="{model_name}",version="{model_version}"}}'
    elif metric_type == "bucket":
        return f'{metric_family}_{metric_type}{{model="{model_name}",version="{model_version}",le="{le}"}}'
    else:
        return None


class TestHistogramMetrics(tu.TestResultCollector):
    def setUp(self):
        self.tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

    def get_metrics(self):
        r = requests.get(f"http://{self.tritonserver_ipaddr}:8002/metrics")
        r.raise_for_status()
        return r.text

    def get_histogram_metrics(self, metric_family: str):

        pattern = f"^{metric_family}.*"
        histogram_dict = {}

        metrics = self.get_metrics()

        matches = re.findall(pattern, metrics, re.MULTILINE)

        for match in matches:
            key, value = match.rsplit(" ")
            histogram_dict[key] = int(value)

        return histogram_dict

    def async_stream_infer(self, model_name, inputs, outputs, responses_per_req):
        with grpcclient.InferenceServerClient(url="localhost:8001") as triton_client:

            def callback(user_data, result, error):
                if error:
                    user_data.append(error)
                else:
                    user_data.append(result)

            user_data = []

            triton_client.start_stream(callback=partial(callback, user_data))
            triton_client.async_stream_infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
            )

        self.assertEqual(len(user_data), responses_per_req)

        for i in range(len(user_data)):

            self.assertNotIsInstance(
                user_data[i], InferenceServerException, user_data[i]
            )

    def test_ensemble_decoupled(self):
        wait_secs = 1
        responses_per_req = 3
        total_reqs = 3
        delta = 0.2

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput("INPUT0", [1], "FP32"))
        inputs.append(grpcclient.InferInput("INPUT1", [1], "UINT8"))
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT"))

        input_data_0 = np.array([wait_secs], np.float32)
        input_data_1 = np.array([responses_per_req], np.uint8)

        inputs[0].set_data_from_numpy(input_data_0)
        inputs[1].set_data_from_numpy(input_data_1)

        for request_num in range(1, total_reqs + 1):
            ensemble_model_name = "ensemble"
            decoupled_model_name = "async_execute_decouple"
            non_decoupled_model_name = "async_execute"
            self.async_stream_infer(
                ensemble_model_name, inputs, outputs, responses_per_req
            )

            histogram_dict = self.get_histogram_metrics(FIRST_RESPONSE_HISTOGRAM)

            def check_existing_metrics(model_name, wait_secs_per_req, delta):
                metric_count = get_histogram_metric_key(
                    FIRST_RESPONSE_HISTOGRAM, model_name, "1", "count"
                )
                metric_sum = get_histogram_metric_key(
                    FIRST_RESPONSE_HISTOGRAM, model_name, "1", "sum"
                )

                self.assertIn(metric_count, histogram_dict)
                self.assertEqual(histogram_dict[metric_count], request_num)

                self.assertIn(metric_sum, histogram_dict)
                self.assertTrue(
                    wait_secs_per_req * MILLIS_PER_SEC * request_num
                    <= histogram_dict[metric_sum]
                    < (wait_secs_per_req + delta) * MILLIS_PER_SEC * request_num
                )

            check_existing_metrics(ensemble_model_name, 2 * wait_secs, 2 * delta)

            check_existing_metrics(decoupled_model_name, wait_secs, delta)

            non_decoupled_model_count = get_histogram_metric_key(
                FIRST_RESPONSE_HISTOGRAM, non_decoupled_model_name, "1", "count"
            )
            non_decoupled_model_sum = get_histogram_metric_key(
                FIRST_RESPONSE_HISTOGRAM, non_decoupled_model_name, "1", "sum"
            )
            self.assertNotIn(non_decoupled_model_count, histogram_dict)
            self.assertNotIn(non_decoupled_model_sum, histogram_dict)

    def test_buckets_override(self):
        model_name = "async_execute_decouple"
        metrics = self.get_metrics()
        override_buckets = [x for x in os.environ.get("OVERRIDE_BUCKETS").split(",")]

        self.assertEqual(
            metrics.count(FIRST_RESPONSE_HISTOGRAM + "_bucket"), len(override_buckets)
        )
        for le in override_buckets:
            bucket_key = get_histogram_metric_key(
                FIRST_RESPONSE_HISTOGRAM, model_name, "1", "bucket", le
            )
            self.assertIn(bucket_key, metrics)


if __name__ == "__main__":
    unittest.main()
