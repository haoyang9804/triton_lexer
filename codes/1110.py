import sys

sys.path.append("../common")

import json
import sys
import unittest

import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from google.protobuf import json_format
from tritonclient.utils import InferenceServerException


class TraceEndpointTest(tu.TestResultCollector):
    def tearDown(self):

        clear_settings = {
            "trace_level": None,
            "trace_rate": None,
            "trace_count": None,
            "log_frequency": None,
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        triton_client.update_trace_settings(
            model_name="simple", settings=clear_settings
        )
        triton_client.update_trace_settings(model_name=None, settings=clear_settings)

    def check_server_initial_state(self):

        initial_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["TIMESTAMPS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0",
            "trace_mode": "triton",
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        self.assertEqual(
            initial_settings, triton_client.get_trace_settings(model_name="simple")
        )
        self.assertEqual(initial_settings, triton_client.get_trace_settings())

    def test_http_get_settings(self):

        initial_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["TIMESTAMPS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0",
            "trace_mode": "triton",
        }
        triton_client = httpclient.InferenceServerClient("localhost:8000")
        self.assertEqual(
            initial_settings,
            triton_client.get_trace_settings(model_name="simple"),
            "Unexpected initial model trace settings",
        )
        self.assertEqual(
            initial_settings,
            triton_client.get_trace_settings(),
            "Unexpected initial global settings",
        )
        try:
            triton_client.get_trace_settings(model_name="does-not-exist")
        except Exception as ex:
            self.assertIn(
                "Request for unknown model : does-not-exist",
                ex.message(),
            )

    def test_grpc_get_settings(self):

        initial_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "trace_file": {"value": ["global_unittest.log"]},
                        "trace_level": {"value": ["TIMESTAMPS"]},
                        "trace_rate": {"value": ["1"]},
                        "trace_count": {"value": ["-1"]},
                        "trace_mode": {"value": ["triton"]},
                        "log_frequency": {"value": ["0"]},
                    }
                }
            ),
            initial_settings,
        )

        triton_client = grpcclient.InferenceServerClient("localhost:8001")
        self.assertEqual(
            initial_settings,
            triton_client.get_trace_settings(model_name="simple"),
            "Unexpected initial model trace settings",
        )
        self.assertEqual(
            initial_settings,
            triton_client.get_trace_settings(),
            "Unexpected initial global settings",
        )
        try:
            triton_client.get_trace_settings(model_name="does-not-exist")
        except Exception as ex:
            self.assertIn(
                "Request for unknown model : does-not-exist",
                ex.message(),
            )

    def test_http_update_settings(self):

        self.check_server_initial_state()

        expected_first_model_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["TIMESTAMPS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0",
            "trace_mode": "triton",
        }
        expected_first_model_response = {
            "error": "trace file location can not be updated through network protocol"
        }
        expected_second_model_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["TIMESTAMPS", "TENSORS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0",
            "trace_mode": "triton",
        }
        expected_global_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["TIMESTAMPS", "TENSORS"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0",
            "trace_mode": "triton",
        }

        model_update_settings = {"trace_file": "model.log"}
        global_update_settings = {
            "trace_level": ["TIMESTAMPS", "TENSORS"],
        }

        triton_client = httpclient.InferenceServerClient("localhost:8000")
        with self.assertRaisesRegex(
            InferenceServerException, expected_first_model_response["error"]
        ) as e:
            triton_client.update_trace_settings(
                model_name="simple", settings=model_update_settings
            )
        self.assertEqual(
            expected_first_model_settings,
            triton_client.get_trace_settings(model_name="simple"),
            "Unexpected model trace settings after global update",
        )

        self.assertEqual(
            expected_global_settings,
            triton_client.update_trace_settings(settings=global_update_settings),
            "Unexpected updated global settings",
        )
        self.assertEqual(
            expected_second_model_settings,
            triton_client.get_trace_settings(model_name="simple"),
            "Unexpected model trace settings after global update",
        )
        try:
            triton_client.update_trace_settings(
                model_name="does-not-exist", settings=model_update_settings
            )
        except Exception as ex:
            self.assertIn(
                "Request for unknown model : does-not-exist",
                ex.message(),
            )

    def test_grpc_update_settings(self):

        self.check_server_initial_state()

        expected_first_model_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "trace_file": {"value": ["global_unittest.log"]},
                        "trace_level": {"value": ["TIMESTAMPS"]},
                        "trace_rate": {"value": ["1"]},
                        "trace_count": {"value": ["-1"]},
                        "log_frequency": {"value": ["0"]},
                        "trace_mode": {"value": ["triton"]},
                    }
                }
            ),
            expected_first_model_settings,
        )

        expected_second_model_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "trace_file": {"value": ["global_unittest.log"]},
                        "trace_level": {"value": ["TIMESTAMPS", "TENSORS"]},
                        "trace_rate": {"value": ["1"]},
                        "trace_count": {"value": ["-1"]},
                        "log_frequency": {"value": ["0"]},
                        "trace_mode": {"value": ["triton"]},
                    }
                }
            ),
            expected_second_model_settings,
        )

        expected_global_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "trace_file": {"value": ["global_unittest.log"]},
                        "trace_level": {"value": ["TIMESTAMPS", "TENSORS"]},
                        "trace_rate": {"value": ["1"]},
                        "trace_count": {"value": ["-1"]},
                        "log_frequency": {"value": ["0"]},
                        "trace_mode": {"value": ["triton"]},
                    }
                }
            ),
            expected_global_settings,
        )

        model_update_settings = {"trace_file": "model.log"}
        global_update_settings = {
            "trace_level": ["TIMESTAMPS", "TENSORS"],
        }

        triton_client = grpcclient.InferenceServerClient("localhost:8001")

        self.assertEqual(
            expected_global_settings,
            triton_client.update_trace_settings(settings=global_update_settings),
            "Unexpected updated global settings",
        )
        self.assertEqual(
            expected_second_model_settings,
            triton_client.get_trace_settings(model_name="simple"),
            "Unexpected model trace settings after global update",
        )
        try:
            triton_client.update_trace_settings(
                model_name="does-not-exist", settings=model_update_settings
            )
        except Exception as ex:
            self.assertIn(
                "Request for unknown model : does-not-exist",
                ex.message(),
            )

    def test_http_clear_settings(self):

        self.check_server_initial_state()

        triton_client = httpclient.InferenceServerClient("localhost:8000")
        triton_client.update_trace_settings(
            model_name="simple", settings={"trace_rate": "12", "log_frequency": "34"}
        )
        triton_client.update_trace_settings(
            settings={"trace_rate": "56", "trace_count": "78", "trace_level": ["OFF"]}
        )

        expected_global_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["OFF"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "0",
            "trace_mode": "triton",
        }
        expected_first_model_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["OFF"],
            "trace_rate": "12",
            "trace_count": "-1",
            "log_frequency": "34",
            "trace_mode": "triton",
        }
        expected_second_model_settings = {
            "trace_file": "global_unittest.log",
            "trace_level": ["OFF"],
            "trace_rate": "1",
            "trace_count": "-1",
            "log_frequency": "34",
            "trace_mode": "triton",
        }
        global_clear_settings = {"trace_rate": None, "trace_count": None}
        model_clear_settings = {"trace_rate": None, "trace_level": None}

        self.assertEqual(
            expected_global_settings,
            triton_client.update_trace_settings(settings=global_clear_settings),
            "Unexpected cleared global trace settings",
        )
        self.assertEqual(
            expected_first_model_settings,
            triton_client.get_trace_settings(model_name="simple"),
            "Unexpected model trace settings after global clear",
        )
        self.assertEqual(
            expected_second_model_settings,
            triton_client.update_trace_settings(
                model_name="simple", settings=model_clear_settings
            ),
            "Unexpected model trace settings after model clear",
        )
        self.assertEqual(
            expected_global_settings,
            triton_client.get_trace_settings(),
            "Unexpected global trace settings after model clear",
        )

    def test_grpc_clear_settings(self):

        self.check_server_initial_state()

        triton_client = grpcclient.InferenceServerClient("localhost:8001")
        triton_client.update_trace_settings(
            model_name="simple", settings={"trace_rate": "12", "log_frequency": "34"}
        )
        triton_client.update_trace_settings(
            settings={"trace_rate": "56", "trace_count": "78", "trace_level": ["OFF"]}
        )

        expected_global_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "trace_file": {"value": ["global_unittest.log"]},
                        "trace_level": {"value": ["OFF"]},
                        "trace_mode": {"value": ["triton"]},
                        "trace_rate": {"value": ["1"]},
                        "trace_count": {"value": ["-1"]},
                        "log_frequency": {"value": ["0"]},
                    }
                }
            ),
            expected_global_settings,
        )
        expected_first_model_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "trace_file": {"value": ["global_unittest.log"]},
                        "trace_level": {"value": ["OFF"]},
                        "trace_rate": {"value": ["12"]},
                        "trace_count": {"value": ["-1"]},
                        "log_frequency": {"value": ["34"]},
                        "trace_mode": {"value": ["triton"]},
                    }
                }
            ),
            expected_first_model_settings,
        )
        expected_second_model_settings = grpcclient.service_pb2.TraceSettingResponse()
        json_format.Parse(
            json.dumps(
                {
                    "settings": {
                        "trace_file": {"value": ["global_unittest.log"]},
                        "trace_level": {"value": ["OFF"]},
                        "trace_rate": {"value": ["1"]},
                        "trace_count": {"value": ["-1"]},
                        "log_frequency": {"value": ["34"]},
                        "trace_mode": {"value": ["triton"]},
                    }
                }
            ),
            expected_second_model_settings,
        )

        global_clear_settings = {"trace_rate": None, "trace_count": None}
        model_clear_settings = {"trace_rate": None, "trace_level": None}

        self.assertEqual(
            expected_global_settings,
            triton_client.update_trace_settings(settings=global_clear_settings),
            "Unexpected cleared global trace settings",
        )
        self.assertEqual(
            expected_first_model_settings,
            triton_client.get_trace_settings(model_name="simple"),
            "Unexpected model trace settings after global clear",
        )
        self.assertEqual(
            expected_second_model_settings,
            triton_client.update_trace_settings(
                model_name="simple", settings=model_clear_settings
            ),
            "Unexpected model trace settings after model clear",
        )
        self.assertEqual(
            expected_global_settings,
            triton_client.get_trace_settings(),
            "Unexpected global trace settings after model clear",
        )


if __name__ == "__main__":
    unittest.main()
