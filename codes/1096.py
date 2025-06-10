import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class TestResponseStatistics(unittest.TestCase):
    def setUp(self):
        self._model_name = "set_by_test_case"
        self._min_infer_delay_ns = 0
        self._min_output_delay_ns = 0
        self._min_cancel_delay_ns = 0
        self._number_of_fail_responses = 0
        self._number_of_empty_responses = 0
        self._statistics_counts = []
        self._grpc_client = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        self._http_client = httpclient.InferenceServerClient("localhost:8000")

    def _generate_streaming_callback_and_response_pair(self):

        response = []

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    def _stream_infer(self, number_of_responses, cancel_at_response_size=None):
        callback, responses = self._generate_streaming_callback_and_response_pair()
        self._grpc_client.start_stream(callback)
        input_data = np.array([number_of_responses], dtype=np.int32)
        inputs = [grpcclient.InferInput("IN", input_data.shape, "INT32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput("OUT")]
        self._grpc_client.async_stream_infer(
            model_name=self._model_name, inputs=inputs, outputs=outputs
        )
        if cancel_at_response_size is None:

            while len(responses) < (
                number_of_responses - self._number_of_empty_responses
            ):
                time.sleep(0.1)
            self._grpc_client.stop_stream(cancel_requests=False)
        else:

            while len(responses) < cancel_at_response_size:
                time.sleep(0.1)
            self._grpc_client.stop_stream(cancel_requests=True)
        return responses

    def _update_statistics_counts(
        self, current_index, number_of_responses, cancel_at_index
    ):
        if current_index >= len(self._statistics_counts):
            self._statistics_counts.append(
                {
                    "compute_infer": 0,
                    "compute_output": 0,
                    "success": 0,
                    "fail": 0,
                    "empty_response": 0,
                    "cancel": 0,
                }
            )
        if current_index == cancel_at_index:

            self._statistics_counts[current_index]["cancel"] += 1
        elif (
            current_index
            + self._number_of_fail_responses
            + self._number_of_empty_responses
            < number_of_responses
        ):

            self._statistics_counts[current_index]["compute_infer"] += 1
            self._statistics_counts[current_index]["compute_output"] += 1
            self._statistics_counts[current_index]["success"] += 1
        elif current_index + self._number_of_empty_responses < number_of_responses:

            self._statistics_counts[current_index]["compute_infer"] += 1
            self._statistics_counts[current_index]["compute_output"] += 1
            self._statistics_counts[current_index]["fail"] += 1
        else:

            self._statistics_counts[current_index]["compute_infer"] += 1
            self._statistics_counts[current_index]["empty_response"] += 1

    def _check_statistics_count_and_duration(
        self, response_stats, current_index, stats_name
    ):
        expected_count = self._statistics_counts[current_index][stats_name]
        if stats_name == "compute_infer" or stats_name == "empty_response":
            delay_ns = self._min_infer_delay_ns
        elif stats_name == "compute_output":
            delay_ns = self._min_output_delay_ns
        elif stats_name == "cancel":
            delay_ns = self._min_cancel_delay_ns
        else:
            delay_ns = self._min_infer_delay_ns + self._min_output_delay_ns
        if delay_ns == 0:
            upper_bound_ns = 10000000 * expected_count
            lower_bound_ns = 0
        else:
            upper_bound_ns = 1.1 * delay_ns * expected_count
            lower_bound_ns = 0.9 * delay_ns * expected_count
        stats = response_stats[str(current_index)][stats_name]
        self.assertEqual(stats["count"], expected_count)
        self.assertLessEqual(stats["ns"], upper_bound_ns)
        self.assertGreaterEqual(stats["ns"], lower_bound_ns)

    def _get_response_statistics(self):

        statistics_http = self._http_client.get_inference_statistics(
            model_name=self._model_name
        )
        model_stats_http = statistics_http["model_stats"][0]
        self.assertEqual(model_stats_http["name"], self._model_name)
        response_stats_http = model_stats_http["response_stats"]

        statistics_grpc = self._grpc_client.get_inference_statistics(
            model_name=self._model_name, as_json=True
        )
        model_stats_grpc = statistics_grpc["model_stats"][0]
        self.assertEqual(model_stats_grpc["name"], self._model_name)
        response_stats_grpc = model_stats_grpc["response_stats"]

        self.assertEqual(len(response_stats_http), len(response_stats_grpc))
        for idx, statistics_http in response_stats_http.items():
            self.assertIn(idx, response_stats_grpc)
            statistics_grpc = response_stats_grpc[idx]
            for name, stats_http in statistics_http.items():
                self.assertIn(name, statistics_grpc)
                stats_grpc = statistics_grpc[name]

                stats_grpc["count"] = (
                    int(stats_grpc["count"]) if ("count" in stats_grpc) else 0
                )
                stats_grpc["ns"] = int(stats_grpc["ns"]) if ("ns" in stats_grpc) else 0

                self.assertEqual(stats_http, stats_grpc)
        return response_stats_http

    def _check_response_stats(
        self, responses, number_of_responses, cancel_at_index=None
    ):
        response_stats = self._get_response_statistics()
        self.assertGreaterEqual(len(response_stats), number_of_responses)
        for i in range(number_of_responses):
            self._update_statistics_counts(i, number_of_responses, cancel_at_index)
            self._check_statistics_count_and_duration(
                response_stats, i, "compute_infer"
            )
            self._check_statistics_count_and_duration(
                response_stats, i, "compute_output"
            )
            self._check_statistics_count_and_duration(response_stats, i, "success")
            self._check_statistics_count_and_duration(response_stats, i, "fail")
            self._check_statistics_count_and_duration(
                response_stats, i, "empty_response"
            )
            self._check_statistics_count_and_duration(response_stats, i, "cancel")

    def test_response_statistics(self):
        self._model_name = "square_int32"
        self._min_infer_delay_ns = 400000000
        self._min_output_delay_ns = 200000000
        self._number_of_fail_responses = 2
        self._number_of_empty_responses = 1

        number_of_responses = 4
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)

        number_of_responses = 6
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)

        number_of_responses = 3
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)

    def test_response_statistics_cancel(self):
        self._model_name = "square_int32_slow"
        self._min_infer_delay_ns = 1200000000
        self._min_output_delay_ns = 800000000
        self._min_cancel_delay_ns = 400000000

        number_of_responses = 4
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)

        responses = self._stream_infer(number_of_responses=4, cancel_at_response_size=1)

        min_total_delay_ns = (
            self._min_infer_delay_ns + self._min_output_delay_ns
        ) * 2 + self._min_cancel_delay_ns

        time.sleep(min_total_delay_ns * 1.5 / 1000000000)

        self._check_response_stats(responses, number_of_responses=3, cancel_at_index=2)


if __name__ == "__main__":
    unittest.main()
