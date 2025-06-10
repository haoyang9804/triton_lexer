



























import sys

sys.path.append("../common")

import base64
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import signal
import threading
import time
import unittest
from builtins import range
from functools import partial
from pathlib import Path

import infer_util as iu
import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class LifeCycleTest(tu.TestResultCollector):
    def _infer_success_models(
        self, model_base_names, versions, tensor_shape, swap=False
    ):
        for base_name in model_base_names:
            try:
                model_name = tu.get_model_name(
                    base_name, np.float32, np.float32, np.float32
                )
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    
                    
                    for v in versions:
                        self.assertTrue(
                            triton_client.is_model_ready(model_name, str(v))
                        )

                for v in versions:
                    iu.infer_exact(
                        self,
                        base_name,
                        tensor_shape,
                        1,
                        np.float32,
                        np.float32,
                        np.float32,
                        model_version=v,
                        swap=(swap or (v != 1)),
                    )
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def _infer_success_identity(self, model_base, versions, tensor_dtype, tensor_shape):
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            for v in versions:
                self.assertTrue(
                    triton_client.is_model_ready(
                        tu.get_zero_model_name(model_base, 1, tensor_dtype), str(v)
                    )
                )

            for v in versions:
                iu.infer_zero(
                    self,
                    model_base,
                    1,
                    tensor_dtype,
                    tensor_shape,
                    tensor_shape,
                    use_http=False,
                    use_grpc=True,
                    use_http_json_tensors=False,
                    use_streaming=False,
                )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def _get_client(self, use_grpc=False):
        if use_grpc:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
        else:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
        return triton_client

    def _async_load(self, model_name, use_grpc):
        try:
            triton_client = self._get_client(use_grpc)
            triton_client.load_model(model_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_noexit(self):
        
        
        
        
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            self.assertFalse(triton_client.is_server_live())
            self.assertFalse(triton_client.is_server_ready())
            md = triton_client.get_server_metadata()
            self.assertEqual(os.environ["TRITON_SERVER_VERSION"], md.version)
            self.assertEqual("triton", md.name)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            self.assertFalse(triton_client.is_server_live())
            self.assertFalse(triton_client.is_server_ready())
            md = triton_client.get_server_metadata()
            self.assertEqual(os.environ["TRITON_SERVER_VERSION"], md["version"])
            self.assertEqual("triton", md["name"])
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_modelfail(self):
        
        tensor_shape = (1, 16)

        
        try:
            model_name = tu.get_model_name(
                "libtorch", np.float32, np.float32, np.float32
            )

            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            self.assertTrue(triton_client.is_server_live())
            self.assertFalse(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))

            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            self.assertTrue(triton_client.is_server_live())
            self.assertFalse(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            iu.infer_exact(
                self, "libtorch", tensor_shape, 1, np.float32, np.float32, np.float32
            )
            self.assertTrue(False, "expected error for unavailable model " + model_name)
        except Exception as ex:
            self.assertIn(
                "Request for unknown model: 'libtorch_float32_float32_float32' has no available versions",
                ex.message(),
            )

        
        try:
            for base_name in ["openvino", "onnx"]:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    model_name = tu.get_model_name(
                        base_name, np.float32, np.float32, np.float32
                    )
                    self.assertTrue(triton_client.is_model_ready(model_name, "1"))

                iu.infer_exact(
                    self,
                    base_name,
                    tensor_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    model_version=1,
                )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_modelfail_nostrict(self):
        
        tensor_shape = (1, 16)

        
        try:
            model_name = tu.get_model_name(
                "libtorch", np.float32, np.float32, np.float32
            )

            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))

            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            iu.infer_exact(
                self, "libtorch", tensor_shape, 1, np.float32, np.float32, np.float32
            )
            self.assertTrue(False, "expected error for unavailable model " + model_name)
        except Exception as ex:
            self.assertIn(
                "Request for unknown model: 'libtorch_float32_float32_float32' has no available versions",
                ex.message(),
            )

        
        try:
            for base_name in ["openvino", "onnx"]:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    model_name = tu.get_model_name(
                        base_name, np.float32, np.float32, np.float32
                    )
                    self.assertTrue(triton_client.is_model_ready(model_name, "1"))

                iu.infer_exact(
                    self,
                    base_name,
                    tensor_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    model_version=1,
                )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_no_model_config(self):
        tensor_shape = (1, 16)

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                model_name = tu.get_model_name(
                    "openvino", np.float32, np.float32, np.float32
                )

                
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())

                md = triton_client.get_model_metadata(model_name, "1")
                self.assertTrue(
                    False,
                    "expected model '"
                    + model_name
                    + "' to be ignored due to polling failure",
                )

            except Exception as ex:
                self.assertIn(
                    "Request for unknown model: 'openvino_float32_float32_float32' is not found",
                    ex.message(),
                )

        
        try:
            for base_name in ["libtorch", "onnx"]:
                model_name = tu.get_model_name(
                    base_name, np.float32, np.float32, np.float32
                )
                self.assertTrue(triton_client.is_model_ready(model_name, "1"))

                iu.infer_exact(
                    self,
                    base_name,
                    tensor_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    model_version=1,
                )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_init_error_modelfail(self):
        

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertFalse(triton_client.is_server_ready())

                
                model_names = ["onnx_sequence_int32", "onnx_int32_int32_int32"]
                for model_name in model_names:
                    self.assertFalse(triton_client.is_model_ready(model_name))

            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            
            try:
                for base_name in ["openvino", "libtorch", "onnx"]:
                    model_name = tu.get_model_name(
                        base_name, np.float32, np.float32, np.float32
                    )
                    self.assertTrue(triton_client.is_model_ready(model_name))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            tensor_shape = (1, 16)
            for base_name in ["openvino", "libtorch", "onnx"]:
                iu.infer_exact(
                    self,
                    base_name,
                    tensor_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    model_version=1,
                )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_error_model_no_version(self):
        
        tensor_shape = (1, 16)

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertFalse(triton_client.is_server_ready())

                model_name = tu.get_model_name(
                    "openvino", np.float32, np.float32, np.float32
                )
                self.assertFalse(triton_client.is_model_ready(model_name))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            
            try:
                for base_name in ["libtorch", "onnx"]:
                    model_name = tu.get_model_name(
                        base_name, np.float32, np.float32, np.float32
                    )
                    self.assertTrue(triton_client.is_model_ready(model_name))
                for version in ["1", "3"]:
                    model_name = tu.get_model_name(
                        "plan", np.float32, np.float32, np.float32
                    )
                    self.assertTrue(triton_client.is_model_ready(model_name, version))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            for base_name in ["libtorch", "onnx"]:
                iu.infer_exact(
                    self,
                    base_name,
                    tensor_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=True,
                )
            for version in [1, 3]:
                iu.infer_exact(
                    self,
                    "plan",
                    tensor_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=(version == 3),
                    model_version=version,
                )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            iu.infer_exact(
                self, "openvino", tensor_shape, 1, np.float32, np.float32, np.float32
            )
            self.assertTrue(False, "expected error for unavailable model " + model_name)
        except Exception as ex:
            self.assertIn(
                "Request for unknown model: 'openvino_float32_float32_float32' has no available versions",
                ex.message(),
            )

    def test_parse_ignore_zero_prefixed_version(self):
        tensor_shape = (1, 16)

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())

                model_name = tu.get_model_name(
                    "libtorch", np.float32, np.float32, np.float32
                )
                self.assertTrue(triton_client.is_model_ready(model_name, "1"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            
            iu.infer_exact(
                self,
                "libtorch",
                tensor_shape,
                1,
                np.float32,
                np.float32,
                np.float32,
                swap=False,
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_parse_ignore_non_intergral_version(self):
        tensor_shape = (1, 16)

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())

                model_name = tu.get_model_name(
                    "libtorch", np.float32, np.float32, np.float32
                )
                self.assertTrue(triton_client.is_model_ready(model_name, "1"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        try:
            
            iu.infer_exact(
                self,
                "libtorch",
                tensor_shape,
                1,
                np.float32,
                np.float32,
                np.float32,
                swap=False,
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_model_load_unload(self):
        tensor_shape = (1, 16)
        libtorch_name = tu.get_model_name(
            "libtorch", np.float32, np.float32, np.float32
        )
        onnx_name = tu.get_model_name("onnx", np.float32, np.float32, np.float32)

        
        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "1"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            shutil.copytree(libtorch_name, "models/" + libtorch_name)
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "1"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            iu.infer_exact(
                self,
                "libtorch",
                tensor_shape,
                1,
                np.float32,
                np.float32,
                np.float32,
                swap=True,
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            stats = triton_client.get_inference_statistics(libtorch_name)
            self.assertEqual(len(stats["model_stats"]), 2)
            for idx in range(len(stats["model_stats"])):
                self.assertEqual(stats["model_stats"][idx]["name"], libtorch_name)
                if stats["model_stats"][idx]["version"] == "1":
                    self.assertEqual(
                        stats["model_stats"][idx]["inference_stats"]["success"][
                            "count"
                        ],
                        0,
                    )
                else:
                    self.assertNotEqual(
                        stats["model_stats"][idx]["inference_stats"]["success"][
                            "count"
                        ],
                        0,
                    )

            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            stats = triton_client.get_inference_statistics(libtorch_name)
            self.assertEqual(len(stats.model_stats), 2)
            for idx in range(len(stats.model_stats)):
                self.assertEqual(stats.model_stats[idx].name, libtorch_name)
                if stats.model_stats[idx].version == "1":
                    self.assertEqual(
                        stats.model_stats[idx].inference_stats.success.count, 0
                    )
                else:
                    self.assertNotEqual(
                        stats.model_stats[idx].inference_stats.success.count, 0
                    )

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            shutil.rmtree("models/" + libtorch_name)
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "1"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            iu.infer_exact(
                self,
                "libtorch",
                tensor_shape,
                1,
                np.float32,
                np.float32,
                np.float32,
                swap=True,
            )
            self.assertTrue(
                False, "expected error for unavailable model " + libtorch_name
            )
        except Exception as ex:
            self.assertIn(
                "Request for unknown model: '{}' has no available versions".format(
                    libtorch_name
                ),
                ex.message(),
            )

        
        try:
            shutil.copytree(libtorch_name, "models/" + libtorch_name)
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "1"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "3"))

            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            stats = triton_client.get_inference_statistics(libtorch_name)
            self.assertEqual(len(stats["model_stats"]), 2)
            self.assertEqual(stats["model_stats"][0]["name"], libtorch_name)
            self.assertEqual(stats["model_stats"][1]["name"], libtorch_name)
            self.assertEqual(
                stats["model_stats"][0]["inference_stats"]["success"]["count"], 0
            )
            self.assertEqual(
                stats["model_stats"][1]["inference_stats"]["success"]["count"], 0
            )

            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            stats = triton_client.get_inference_statistics(libtorch_name)
            self.assertEqual(len(stats.model_stats), 2)
            self.assertEqual(stats.model_stats[0].name, libtorch_name)
            self.assertEqual(stats.model_stats[1].name, libtorch_name)
            self.assertEqual(stats.model_stats[0].inference_stats.success.count, 0)
            self.assertEqual(stats.model_stats[1].inference_stats.success.count, 0)

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            shutil.rmtree("models/" + onnx_name)
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertFalse(triton_client.is_model_ready(onnx_name, "1"))
                self.assertFalse(triton_client.is_model_ready(onnx_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            iu.infer_exact(
                self,
                "onnx",
                tensor_shape,
                1,
                np.float32,
                np.float32,
                np.float32,
                swap=True,
            )
            self.assertTrue(False, "expected error for unavailable model " + onnx_name)
        except Exception as ex:
            self.assertIn(
                "Request for unknown model: 'onnx_float32_float32_float32' has no available versions",
                ex.message(),
            )

    def test_dynamic_model_load_unload_disabled(self):
        tensor_shape = (1, 16)
        libtorch_name = tu.get_model_name(
            "libtorch", np.float32, np.float32, np.float32
        )
        onnx_name = tu.get_model_name("onnx", np.float32, np.float32, np.float32)

        
        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "1"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            shutil.copytree(libtorch_name, "models/" + libtorch_name)
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "1"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            iu.infer_exact(
                self,
                "libtorch",
                tensor_shape,
                1,
                np.float32,
                np.float32,
                np.float32,
                swap=True,
            )
            self.assertTrue(
                False, "expected error for unavailable model " + libtorch_name
            )
        except Exception as ex:
            self.assertIn(
                "Request for unknown model: 'libtorch_float32_float32_float32' is not found",
                ex.message(),
            )

        
        
        try:
            shutil.rmtree("models/" + onnx_name)
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "1"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            iu.infer_exact(
                self,
                "onnx",
                tensor_shape,
                1,
                np.float32,
                np.float32,
                np.float32,
                swap=True,
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_version_load_unload(self):
        tensor_shape = (1, 16)
        libtorch_name = tu.get_model_name("libtorch", np.int32, np.int32, np.int32)

        
        
        try:
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "2"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            iu.infer_exact(
                self,
                "libtorch",
                tensor_shape,
                1,
                np.int32,
                np.int32,
                np.int32,
                swap=False,
                model_version=1,
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            stats = triton_client.get_inference_statistics(libtorch_name)
            self.assertEqual(len(stats["model_stats"]), 3)
            for idx in range(len(stats["model_stats"])):
                self.assertEqual(stats["model_stats"][idx]["name"], libtorch_name)
                if stats["model_stats"][idx]["version"] == "1":
                    self.assertNotEqual(
                        stats["model_stats"][idx]["inference_stats"]["success"][
                            "count"
                        ],
                        0,
                    )
                else:
                    self.assertEqual(
                        stats["model_stats"][idx]["inference_stats"]["success"][
                            "count"
                        ],
                        0,
                    )

            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            stats = triton_client.get_inference_statistics(libtorch_name)
            self.assertEqual(len(stats.model_stats), 3)
            for idx in range(len(stats.model_stats)):
                self.assertEqual(stats.model_stats[idx].name, libtorch_name)
                if stats.model_stats[idx].version == "1":
                    self.assertNotEqual(
                        stats.model_stats[idx].inference_stats.success.count, 0
                    )
                else:
                    self.assertEqual(
                        stats.model_stats[idx].inference_stats.success.count, 0
                    )

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            shutil.rmtree("models/" + libtorch_name + "/1")
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "2"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            iu.infer_exact(
                self,
                "libtorch",
                tensor_shape,
                1,
                np.int32,
                np.int32,
                np.int32,
                swap=False,
                model_version=1,
            )
            self.assertTrue(
                False, "expected error for unavailable model " + libtorch_name
            )
        except Exception as ex:
            self.assertIn(
                "Request for unknown model: 'libtorch_int32_int32_int32' version 1 is not at ready state",
                ex.message(),
            )

        
        try:
            shutil.copytree(
                "models/" + libtorch_name + "/2", "models/" + libtorch_name + "/7"
            )
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "2"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "7"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_version_load_unload_disabled(self):
        tensor_shape = (1, 16)
        libtorch_name = tu.get_model_name("libtorch", np.int32, np.int32, np.int32)

        
        
        
        try:
            shutil.copytree(
                "models/" + libtorch_name + "/2", "models/" + libtorch_name + "/7"
            )
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "2"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "7"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        
        try:
            shutil.rmtree("models/" + libtorch_name + "/1")
            time.sleep(5)  
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "2"))
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "3"))
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "7"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            iu.infer_exact(
                self,
                "libtorch",
                tensor_shape,
                1,
                np.int32,
                np.int32,
                np.int32,
                swap=False,
                model_version=1,
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_model_modify(self):
        models_base = ("libtorch", "plan")
        models_shape = ((1, 16), (1, 16))
        models = list()
        for m in models_base:
            models.append(tu.get_model_name(m, np.float32, np.float32, np.float32))

        
        for model_name in models:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        for version in (1, 3):
            for model_name, model_shape in zip(models_base, models_shape):
                try:
                    iu.infer_exact(
                        self,
                        model_name,
                        model_shape,
                        1,
                        np.float32,
                        np.float32,
                        np.float32,
                        swap=(version == 3),
                        model_version=version,
                    )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

        
        for base_name, model_name in zip(models_base, models):
            shutil.copyfile(
                "config.pbtxt.wrong." + base_name,
                "models/" + model_name + "/config.pbtxt",
            )

        time.sleep(5)  
        for model_name in models:
            for model_name, model_shape in zip(models_base, models_shape):
                try:
                    iu.infer_exact(
                        self,
                        model_name,
                        model_shape,
                        1,
                        np.float32,
                        np.float32,
                        np.float32,
                        swap=(version == 3),
                        model_version=version,
                        output0_raw=False,
                    )
                    self.assertTrue(
                        False, "expected error for wrong label for " + model_name
                    )
                except AssertionError as ex:
                    self.assertTrue("'label9" in str(ex) and "!=" in str(ex), str(ex))

        
        
        for base_name, model_name in zip(models_base, models):
            shutil.copyfile(
                "config.pbtxt." + base_name, "models/" + model_name + "/config.pbtxt"
            )

        time.sleep(5)  
        for model_name in models:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        for model_name, model_shape in zip(models_base, models_shape):
            try:
                iu.infer_exact(
                    self,
                    model_name,
                    model_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=False,
                    model_version=1,
                )
                self.assertTrue(
                    False, "expected error for unavailable model " + model_name
                )
            except Exception as ex:
                self.assertIn("Request for unknown model", ex.message())

        
        for model_name, model_shape in zip(models_base, models_shape):
            try:
                iu.infer_exact(
                    self,
                    model_name,
                    model_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=True,
                    model_version=3,
                )
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_dynamic_file_delete(self):
        models_base = ("onnx", "plan")
        models_shape = ((1, 16), (1, 16))
        models = list()
        for m in models_base:
            models.append(tu.get_model_name(m, np.float32, np.float32, np.float32))

        
        for model_name in models:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        for version in (1, 3):
            for model_name, model_shape in zip(models_base, models_shape):
                try:
                    iu.infer_exact(
                        self,
                        model_name,
                        model_shape,
                        1,
                        np.float32,
                        np.float32,
                        np.float32,
                        swap=(version == 3),
                        model_version=version,
                    )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        
        
        for model_name in models:
            os.remove("models/" + model_name + "/config.pbtxt")

        time.sleep(5)  
        for model_name in models:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        for model_name, model_shape in zip(models_base, models_shape):
            try:
                iu.infer_exact(
                    self,
                    model_name,
                    model_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=True,
                    model_version=3,
                )
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

            try:
                iu.infer_exact(
                    self,
                    model_name,
                    model_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=False,
                    model_version=1,
                )
                self.assertTrue(
                    False, "expected error for unavailable model " + model_name
                )
            except Exception as ex:
                self.assertIn("Request for unknown model", ex.message())

    def test_multiple_model_repository_polling(self):
        model_shape = (1, 16)
        libtorch_name = tu.get_model_name(
            "libtorch", np.float32, np.float32, np.float32
        )

        
        
        self._infer_success_models(
            [
                "libtorch",
            ],
            (1,),
            model_shape,
        )
        self._infer_success_models(["openvino", "onnx"], (1, 3), model_shape)

        
        
        shutil.copytree(libtorch_name, "models_0/" + libtorch_name)
        time.sleep(5)  
        try:
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "1"))
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(["openvino", "onnx"], (1, 3), model_shape)

        
        
        
        
        shutil.rmtree("models/" + libtorch_name)
        time.sleep(5)  
        self._infer_success_models(
            ["libtorch", "openvino", "onnx"], (1, 3), model_shape
        )

    def test_multiple_model_repository_control(self):
        
        
        model_shape = (1, 16)
        libtorch_name = tu.get_model_name(
            "libtorch", np.float32, np.float32, np.float32
        )
        model_bases = ["libtorch", "openvino", "onnx"]

        
        for base in model_bases:
            try:
                model_name = tu.get_model_name(base, np.float32, np.float32, np.float32)
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        for base in model_bases:
            try:
                model_name = tu.get_model_name(base, np.float32, np.float32, np.float32)
                triton_client = grpcclient.InferenceServerClient(
                    "localhost:8001", verbose=True
                )
                triton_client.load_model(model_name)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        self._infer_success_models(
            [
                "libtorch",
            ],
            (1,),
            model_shape,
        )
        self._infer_success_models(["openvino", "onnx"], (1, 3), model_shape)

        
        
        
        shutil.copytree(libtorch_name, "models_0/" + libtorch_name)
        self._infer_success_models(
            [
                "libtorch",
            ],
            (1,),
            model_shape,
        )
        self._infer_success_models(["openvino", "onnx"], (1, 3), model_shape)

        
        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.load_model(libtorch_name)
        except Exception as ex:
            self.assertIn("failed to load '{}'".format(libtorch_name), ex.message())

        try:
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                
                
                self.assertTrue(triton_client.is_model_ready(libtorch_name, "1"))
                
                
                self.assertFalse(triton_client.is_model_ready(libtorch_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(["openvino", "onnx"], (1, 3), model_shape)

        
        
        
        
        shutil.rmtree("models/" + libtorch_name)
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            
            triton_client.unload_model(libtorch_name)
            
            triton_client.load_model(libtorch_name)
        except Exception as ex:
            self.assertIn("failed to load '{}'".format(libtorch_name), ex.message())

        self._infer_success_models(
            ["libtorch", "openvino", "onnx"], (1, 3), model_shape
        )

    def test_model_control(self):
        model_shape = (1, 16)
        onnx_name = tu.get_model_name("onnx", np.float32, np.float32, np.float32)

        ensemble_prefix = "simple_"
        ensemble_name = ensemble_prefix + onnx_name

        
        for model_name in (onnx_name, ensemble_name):
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                triton_client.load_model("unknown_model")
                self.assertTrue(False, "expected unknown model failure")
            except Exception as ex:
                self.assertIn(
                    "failed to load 'unknown_model', failed to poll from model repository",
                    ex.message(),
                )

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.load_model(ensemble_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(
            [
                "onnx",
            ],
            (1, 3),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_onnx",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )

        
        
        
        for model_name in (onnx_name,):
            os.remove("models/" + model_name + "/config.pbtxt")

        self._infer_success_models(
            [
                "onnx",
            ],
            (1, 3),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_onnx",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )

        
        for model_name in (onnx_name, ensemble_name):
            try:
                triton_client = grpcclient.InferenceServerClient(
                    "localhost:8001", verbose=True
                )
                triton_client.load_model(model_name)
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(
            [
                "onnx",
            ],
            (3,),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_onnx",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )

        for model_name in (onnx_name,):
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                triton_client.unload_model("unknown_model")
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.unload_model(onnx_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        for model_name in (onnx_name, ensemble_name):
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.unload_model(ensemble_name)
            triton_client.load_model(onnx_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(
            [
                "onnx",
            ],
            (3,),
            model_shape,
        )

        try:
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(ensemble_name, "1"))
                self.assertFalse(triton_client.is_model_ready(ensemble_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_model_control_fail(self):
        model_name = tu.get_model_name("onnx", np.float32, np.float32, np.float32)

        
        try:
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                self.assertFalse(triton_client.is_model_ready(model_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.load_model(model_name)
            self.assertTrue(False, "expecting load failure")
        except InferenceServerException as ex:
            self.assertIn("load failed for model '{}'".format(model_name), ex.message())

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.load_model(model_name)
            self.assertTrue(False, "expecting load failure")
        except InferenceServerException as ex:
            self.assertIn("load failed for model '{}'".format(model_name), ex.message())

    def test_model_control_ensemble(self):
        model_shape = (1, 16)
        onnx_name = tu.get_model_name("onnx", np.float32, np.float32, np.float32)

        ensemble_prefix = "simple_"
        ensemble_name = ensemble_prefix + onnx_name

        
        for model_name in (onnx_name, ensemble_name):
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.load_model(ensemble_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(
            [
                "onnx",
            ],
            (1, 3),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_onnx",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.unload_model(ensemble_name, unload_dependents=True)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        for model_name in (onnx_name, ensemble_name):
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.load_model(ensemble_name)
            triton_client.unload_model(ensemble_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(
            [
                "onnx",
            ],
            (1, 3),
            model_shape,
        )

        try:
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(ensemble_name, "1"))
                self.assertFalse(triton_client.is_model_ready(ensemble_name, "3"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "1"))
                self.assertTrue(triton_client.is_model_ready(onnx_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_load_same_model_different_platform(self):
        model_shape = (1, 16)
        model_name = tu.get_model_name("simple", np.float32, np.float32, np.float32)

        
        use_grpc = "TRITONSERVER_USE_GRPC" in os.environ

        
        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
            self.assertTrue(triton_client.is_model_ready(model_name, "3"))
            if use_grpc:
                metadata = triton_client.get_model_metadata(model_name, as_json=True)
            else:
                metadata = triton_client.get_model_metadata(model_name)
            self.assertEqual(metadata["platform"], "tensorrt_plan")
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_models(
            [
                "simple",
            ],
            (
                1,
                3,
            ),
            model_shape,
        )

        
        shutil.rmtree("models/" + model_name)
        shutil.copytree(model_name, "models/" + model_name)

        
        try:
            triton_client = self._get_client(use_grpc)
            triton_client.load_model(model_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
            self.assertTrue(triton_client.is_model_ready(model_name, "3"))
            if use_grpc:
                metadata = triton_client.get_model_metadata(model_name, as_json=True)
            else:
                metadata = triton_client.get_model_metadata(model_name)
            self.assertEqual(metadata["platform"], "pytorch_libtorch")
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_models(
            [
                "simple",
            ],
            (
                1,
                3,
            ),
            model_shape,
        )

    def test_model_availability_on_reload(self):
        model_name = "identity_zero_1_int32"
        model_base = "identity"
        model_shape = (16,)

        
        use_grpc = "TRITONSERVER_USE_GRPC" in os.environ

        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

        
        os.mkdir("models/" + model_name + "/2")

        
        
        
        thread = threading.Thread(target=self._async_load, args=(model_name, use_grpc))
        thread.start()
        
        time.sleep(3)
        load_start = time.time()

        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            load_end = time.time()
            self.assertTrue(
                (load_end - load_start) < 5,
                "server was waiting unexpectedly, waited {}".format(
                    (load_end - load_start)
                ),
            )
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

        thread.join()
        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))
            self.assertTrue(triton_client.is_model_ready(model_name, "2"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (2,), np.int32, model_shape)

    def test_model_availability_on_reload_2(self):
        model_name = "identity_zero_1_int32"
        model_base = "identity"
        model_shape = (16,)

        
        use_grpc = "TRITONSERVER_USE_GRPC" in os.environ

        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

        
        shutil.copyfile("config.pbtxt.v2", "models/" + model_name + "/config.pbtxt")

        
        
        
        thread = threading.Thread(target=self._async_load, args=(model_name, use_grpc))
        thread.start()
        
        time.sleep(3)
        load_start = time.time()

        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            load_end = time.time()
            self.assertTrue(
                (load_end - load_start) < 5,
                "server was waiting unexpectedly, waited {}".format(
                    (load_end - load_start)
                ),
            )
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

        thread.join()
        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertFalse(triton_client.is_model_ready(model_name, "1"))
            self.assertTrue(triton_client.is_model_ready(model_name, "2"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (2,), np.int32, model_shape)

    def test_model_availability_on_reload_3(self):
        model_name = "identity_zero_1_int32"
        model_base = "identity"
        model_shape = (16,)

        
        use_grpc = "TRITONSERVER_USE_GRPC" in os.environ

        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

        
        shutil.copyfile("config.pbtxt.new", "models/" + model_name + "/config.pbtxt")

        
        
        thread = threading.Thread(target=self._async_load, args=(model_name, use_grpc))
        thread.start()
        
        time.sleep(3)
        load_start = time.time()

        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            load_end = time.time()
            self.assertTrue(
                (load_end - load_start) < 5,
                "server was waiting unexpectedly, waited {}".format(
                    (load_end - load_start)
                ),
            )
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

        thread.join()
        
        try:
            triton_client = self._get_client(use_grpc)
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

    def test_model_reload_fail(self):
        model_name = "identity_zero_1_int32"
        model_base = "identity"
        model_shape = (16,)

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

        
        shutil.copyfile("config.pbtxt.v2.gpu", "models/" + model_name + "/config.pbtxt")

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.load_model(model_name)
            self.assertTrue(False, "expecting load failure")
        except Exception as ex:
            self.assertIn(
                "version 2 is at UNAVAILABLE state: Internal: GPU instances not supported",
                ex.message(),
            )

        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
            self.assertFalse(triton_client.is_model_ready(model_name, "2"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        self._infer_success_identity(model_base, (1,), np.int32, model_shape)

    def test_multiple_model_repository_control_startup_models(self):
        model_shape = (1, 16)
        onnx_name = tu.get_model_name("onnx", np.float32, np.float32, np.float32)
        plan_name = tu.get_model_name("plan", np.float32, np.float32, np.float32)

        ensemble_prefix = "simple_"
        onnx_ensemble_name = ensemble_prefix + onnx_name
        plan_ensemble_name = ensemble_prefix + plan_name

        
        for base in ("libtorch",):
            model_name = tu.get_model_name(base, np.float32, np.float32, np.float32)
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        self._infer_success_models(
            [
                "onnx",
            ],
            (1, 3),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_onnx",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )
        self._infer_success_models(
            [
                "plan",
            ],
            (1, 3),
            model_shape,
        )

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                triton_client.load_model("unknown_model")
                self.assertTrue(False, "expected unknown model failure")
            except Exception as ex:
                self.assertIn(
                    "failed to load 'unknown_model', failed to poll from model repository",
                    ex.message(),
                )

        
        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.load_model(plan_ensemble_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(
            [
                "plan",
            ],
            (1, 3),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_plan",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )

        
        
        
        os.remove("models/" + onnx_name + "/config.pbtxt")

        self._infer_success_models(
            [
                "plan",
            ],
            (1, 3),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_plan",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )

        
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            triton_client.load_model(onnx_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(
            [
                "onnx",
            ],
            (3,),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_onnx",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )

        try:
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(onnx_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            try:
                triton_client.unload_model("unknown_model")
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.unload_model(onnx_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        for model_name in [onnx_name, onnx_ensemble_name]:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            triton_client.unload_model(onnx_ensemble_name)
            triton_client.load_model(onnx_name)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        self._infer_success_models(
            [
                "onnx",
            ],
            (3,),
            model_shape,
        )
        self._infer_success_models(
            [
                "plan",
            ],
            (1, 3),
            model_shape,
        )
        self._infer_success_models(
            [
                "simple_plan",
            ],
            (1, 3),
            model_shape,
            swap=True,
        )

        try:
            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                self.assertTrue(triton_client.is_server_live())
                self.assertTrue(triton_client.is_server_ready())
                self.assertFalse(triton_client.is_model_ready(onnx_ensemble_name, "1"))
                self.assertFalse(triton_client.is_model_ready(onnx_ensemble_name, "3"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_model_repository_index(self):
        
        
        tensor_shape = (1, 16)
        model_bases = ["plan", "libtorch", "simple_libtorch"]

        
        
        
        
        for model_base in model_bases:
            try:
                model_name = tu.get_model_name(
                    model_base, np.float32, np.float32, np.float32
                )
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(triton_client.is_model_ready(model_name))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        
        model_bases.append("simple_plan")
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
            index = triton_client.get_model_repository_index()
            indexed = list()
            self.assertEqual(len(index), 8)
            for i in index:
                indexed.append(i["name"])
                if i["name"] == "onnx_float32_float32_float32":
                    self.assertEqual(i["state"], "UNAVAILABLE")
                    self.assertEqual(
                        i["reason"], "model appears in two or more repositories"
                    )
            for model_base in model_bases:
                model_name = tu.get_model_name(
                    model_base, np.float32, np.float32, np.float32
                )
                self.assertTrue(model_name in indexed)

            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            index = triton_client.get_model_repository_index()
            indexed = list()
            self.assertEqual(len(index.models), 8)
            for i in index.models:
                indexed.append(i.name)
                if i.name == "onnx_float32_float32_float32":
                    self.assertEqual(i.state, "UNAVAILABLE")
                    self.assertEqual(
                        i.reason, "model appears in two or more repositories"
                    )
            for model_base in model_bases:
                model_name = tu.get_model_name(
                    model_base, np.float32, np.float32, np.float32
                )
                self.assertTrue(model_name in indexed)

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_config_override(self):
        model_shape = (1, 16)

        for triton_client in (
            httpclient.InferenceServerClient("localhost:8000", verbose=True),
            grpcclient.InferenceServerClient("localhost:8001", verbose=True),
        ):
            for base in (("onnx", "onnxruntime"),):
                model_name = tu.get_model_name(
                    base[0], np.float32, np.float32, np.float32
                )
                try:
                    self.assertTrue(triton_client.is_server_live())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

                
                
                try:
                    triton_client.load_model(model_name)
                    self.assertTrue(
                        False, "expected fail to load '{}'".format(model_name)
                    )
                except Exception as ex:
                    self.assertIn(
                        "load failed for model '{}'".format(model_name), ex.message()
                    )

                
                try:
                    triton_client.load_model(
                        model_name,
                        config=.format(
                            backend=base[1]
                        ),
                    )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                self.assertTrue(triton_client.is_model_ready(model_name, "2"))
                self.assertFalse(triton_client.is_model_ready(model_name, "3"))

                
                self._infer_success_models(
                    [
                        base[0],
                    ],
                    (2,),
                    model_shape,
                )

                
                
                try:
                    triton_client.load_model(model_name)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                self.assertTrue(triton_client.is_model_ready(model_name, "2"))
                self.assertFalse(triton_client.is_model_ready(model_name, "3"))

                
                try:
                    triton_client.unload_model(model_name)
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

    def test_file_override(self):
        model_shape = (1, 16)
        override_base = "override_model"

        for base in (("onnx", "onnxruntime"),):
            model_name = tu.get_model_name(base[0], np.float32, np.float32, np.float32)
            override_model_name = tu.get_model_name(
                override_base, np.float32, np.float32, np.float32
            )

            
            with open("models/{}/3/model.{}".format(model_name, base[0]), "rb") as f:
                file_content = f.read()

            for triton_client in (
                httpclient.InferenceServerClient("localhost:8000", verbose=True),
                grpcclient.InferenceServerClient("localhost:8001", verbose=True),
            ):
                try:
                    self.assertTrue(triton_client.is_server_live())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "3"))
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

                
                
                
                
                try:
                    triton_client.load_model(
                        model_name, files={"file:1/model.onnx": file_content}
                    )
                    self.assertTrue(False, "expected error on missing override config")
                except InferenceServerException as ex:
                    
                    self.assertIn(
                        "failed to load '{}', failed to poll from model repository".format(
                            model_name
                        ),
                        ex.message(),
                    )

                
                
                self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                self.assertTrue(triton_client.is_model_ready(model_name, "3"))

                self._infer_success_models(
                    [
                        base[0],
                    ],
                    (3,),
                    model_shape,
                )

                
                
                try:
                    triton_client.load_model(
                        override_model_name,
                        config=.format(backend=base[1]),
                        files={"file:1/model.onnx": file_content},
                    )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

                
                
                self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                self.assertTrue(triton_client.is_model_ready(model_name, "3"))
                self._infer_success_models(
                    [
                        base[0],
                    ],
                    (3,),
                    model_shape,
                )

                
                self.assertTrue(triton_client.is_model_ready(override_model_name, "1"))
                self.assertFalse(triton_client.is_model_ready(override_model_name, "2"))
                self.assertFalse(triton_client.is_model_ready(override_model_name, "3"))
                self._infer_success_models(
                    [
                        override_base,
                    ],
                    (1,),
                    model_shape,
                    swap=True,
                )

                
                
                try:
                    triton_client.load_model(
                        model_name,
                        config=.format(backend=base[1]),
                        files={"file:1/model.onnx": file_content},
                    )
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))

                
                
                self.assertTrue(triton_client.is_model_ready(model_name, "1"))
                self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                self.assertFalse(triton_client.is_model_ready(model_name, "3"))
                self._infer_success_models(
                    [
                        base[0],
                    ],
                    (1,),
                    model_shape,
                    swap=True,
                )

                
                self.assertTrue(triton_client.is_model_ready(override_model_name, "1"))
                self.assertFalse(triton_client.is_model_ready(override_model_name, "2"))
                self.assertFalse(triton_client.is_model_ready(override_model_name, "3"))
                self._infer_success_models(
                    [
                        override_base,
                    ],
                    (1,),
                    model_shape,
                    swap=True,
                )

                
                try:
                    
                    
                    triton_client.unload_model(model_name)
                    triton_client.load_model(model_name)
                    triton_client.unload_model(override_model_name)
                except Exception as ex:
                    self.assertTrue(False, "unexpected error {}".format(ex))
                self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                self.assertTrue(triton_client.is_model_ready(model_name, "3"))
                self._infer_success_models(
                    [
                        base[0],
                    ],
                    (3,),
                    model_shape,
                )

    
    
    def test_file_override_security(self):
        
        
        
        model_basepath = "/tmp/folderXXXXXX"
        if os.path.exists(model_basepath) and os.path.isdir(model_basepath):
            shutil.rmtree(model_basepath)
        os.makedirs(model_basepath)

        
        
        root_home_dir = "/root"

        
        escape_dir_rel = os.path.join("..", "..", "root")
        escape_dir_full = os.path.join(model_basepath, escape_dir_rel)
        self.assertEqual(os.path.abspath(escape_dir_full), root_home_dir)

        new_file_rel = os.path.join(escape_dir_rel, "new_dir", "test.txt")
        self.assertFalse(os.path.exists(os.path.join(model_basepath, new_file_rel)))
        existing_file_rel = os.path.join(escape_dir_rel, ".bashrc")
        self.assertTrue(os.path.exists(os.path.join(model_basepath, existing_file_rel)))

        
        
        
        escape_dir_symlink_rel = os.path.join("..", "escape_symlink")
        escape_dir_symlink_full = "/tmp/escape_symlink"
        self.assertEqual(
            os.path.abspath(os.path.join(model_basepath, escape_dir_symlink_rel)),
            escape_dir_symlink_full,
        )
        if os.path.exists(escape_dir_symlink_full):
            os.unlink(escape_dir_symlink_full)
        os.symlink(root_home_dir, escape_dir_symlink_full)
        self.assertTrue(os.path.abspath(escape_dir_symlink_full), root_home_dir)

        symlink_new_file_rel = os.path.join(
            escape_dir_symlink_rel, "new_dir", "test.txt"
        )
        self.assertFalse(
            os.path.exists(os.path.join(model_basepath, symlink_new_file_rel))
        )
        symlink_existing_file_rel = os.path.join(escape_dir_symlink_rel, ".bashrc")
        self.assertTrue(
            os.path.exists(os.path.join(model_basepath, symlink_existing_file_rel))
        )

        
        new_contents = "This shouldn't exist"
        new_contents_b64 = base64.b64encode(new_contents.encode())

        new_files = [new_file_rel, symlink_new_file_rel]
        existing_files = [existing_file_rel, symlink_existing_file_rel]
        all_files = new_files + existing_files
        for filepath in all_files:
            
            config = json.dumps({"backend": "identity"})
            files = {f"file:{filepath}": new_contents_b64}
            with httpclient.InferenceServerClient("localhost:8000") as client:
                with self.assertRaisesRegex(InferenceServerException, "failed to load"):
                    client.load_model("new_model", config=config, files=files)

        for rel_path in new_files:
            
            self.assertFalse(os.path.exists(os.path.join(model_basepath, rel_path)))

        for rel_path in existing_files:
            
            existing_file = os.path.join(model_basepath, rel_path)
            self.assertTrue(os.path.exists(existing_file))
            with open(existing_file) as f:
                contents = f.read()
                self.assertNotEqual(contents, new_contents)

    def test_shutdown_dynamic(self):
        model_shape = (1, 1)
        input_data = np.ones(shape=(1, 1), dtype=np.float32)

        inputs = [grpcclient.InferInput("INPUT0", model_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        triton_client = grpcclient.InferenceServerClient("localhost:8001", verbose=True)
        model_name = "custom_zero_1_float32"

        
        
        def callback(user_data, result, error):
            if error:
                user_data.append(error)
            else:
                user_data.append(result)

        
        
        
        
        request_count = 6
        async_results = []
        for _ in range(request_count):
            triton_client.async_infer(
                model_name, inputs, partial(callback, async_results)
            )
        time.sleep(1)

        
        os.kill(int(os.environ["SERVER_PID"]), signal.SIGINT)
        time.sleep(0.5)

        
        try:
            triton_client.infer(model_name, inputs)
            self.assertTrue(False, "expected error for new inference during shutdown")
        except InferenceServerException as ex:
            self.assertIn(
                "failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:8001: "
                + "Failed to connect to remote host: connect: Connection refused (111)",
                ex.message(),
            )

        
        time_out = 30
        while (len(async_results) < request_count) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)

        
        for result in async_results:
            if type(result) == InferenceServerException:
                raise result
            output_data = result.as_numpy("OUTPUT0")
            np.testing.assert_allclose(
                output_data, input_data, err_msg="Inference result is not correct"
            )

    def test_shutdown_sequence(self):
        model_shape = (1, 1)
        input_data = np.ones(shape=(1, 1), dtype=np.int32)

        inputs = [grpcclient.InferInput("INPUT", model_shape, "INT32")]
        inputs[0].set_data_from_numpy(input_data)

        triton_client = grpcclient.InferenceServerClient("localhost:8001", verbose=True)
        model_name = "custom_sequence_int32"

        
        
        def callback(user_data, result, error):
            if error:
                user_data.append(error)
            else:
                user_data.append(result)

        
        request_count = 2
        async_results = []
        for i in range(request_count):
            triton_client.async_infer(
                model_name,
                inputs,
                partial(callback, async_results),
                sequence_id=(i + 1),
                sequence_start=True,
            )
        time.sleep(1)

        
        os.kill(int(os.environ["SERVER_PID"]), signal.SIGINT)
        time.sleep(0.5)

        
        
        try:
            triton_client.infer(
                model_name, inputs, sequence_id=request_count, sequence_start=True
            )
            self.assertTrue(False, "expected error for new inference during shutdown")
        except InferenceServerException as ex:
            
            
            self.assertIn("CANCELLED", ex.message())
        
        try:
            triton_client.infer(model_name, inputs, sequence_id=1, sequence_start=True)
            self.assertTrue(False, "expected error for new inference during shutdown")
        except InferenceServerException as ex:
            self.assertIn(
                "failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:8001: "
                + "Failed to connect to remote host: connect: Connection refused (111)",
                ex.message(),
            )
        
        try:
            triton_client.infer(model_name, inputs, sequence_id=2, sequence_end=True)
            self.assertTrue(False, "expected error for new inference during shutdown")
        except InferenceServerException as ex:
            self.assertIn(
                "failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:8001: "
                + "Failed to connect to remote host: connect: Connection refused (111)",
                ex.message(),
            )

        
        time_out = 30
        while (len(async_results) < request_count) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)

        
        for result in async_results:
            if type(result) == InferenceServerException:
                raise result
            output_data = result.as_numpy("OUTPUT")
            np.testing.assert_allclose(
                output_data, input_data, err_msg="Inference result is not correct"
            )

        
        
        time.sleep(5)

    def test_shutdown_ensemble(self):
        model_shape = (1, 1)
        input_data = np.ones(shape=(1, 1), dtype=np.float32)

        inputs = [grpcclient.InferInput("INPUT0", model_shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        triton_client = grpcclient.InferenceServerClient("localhost:8001", verbose=True)
        model_name = "ensemble_zero_1_float32"

        
        
        def callback(user_data, result, error):
            if error:
                user_data.append(error)
            else:
                user_data.append(result)

        
        
        
        request_count = 1
        async_results = []
        for _ in range(request_count):
            triton_client.async_infer(
                model_name, inputs, partial(callback, async_results)
            )
        time.sleep(1)

        
        os.kill(int(os.environ["SERVER_PID"]), signal.SIGINT)
        time.sleep(0.5)

        
        try:
            triton_client.infer(model_name, inputs)
            self.assertTrue(False, "expected error for new inference during shutdown")
        except InferenceServerException as ex:
            self.assertIn(
                "failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:8001: "
                + "Failed to connect to remote host: connect: Connection refused (111)",
                ex.message(),
            )

        
        time_out = 10
        while (len(async_results) < request_count) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)

        
        for result in async_results:
            if type(result) == InferenceServerException:
                raise result
            output_data = result.as_numpy("OUTPUT0")
            np.testing.assert_allclose(
                output_data, input_data, err_msg="Inference result is not correct"
            )

    def test_load_gpu_limit(self):
        model_name = "cuda_memory_consumer"
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            triton_client.load_model(model_name + "_1")
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            triton_client.load_model(model_name + "_2")
            self.assertTrue(False, "expected error for loading model")
        except Exception as ex:
            self.assertIn("memory limit set for GPU 0 has exceeded", ex.message())

        
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
            triton_client.unload_model(model_name + "_1")
            triton_client.load_model(model_name + "_2")
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_concurrent_model_load_speedup(self):
        
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        
        model_pairs = [
            ["identity_zero_1_int32_1", "identity_zero_1_int32_2"],
            ["python_identity_fp32_1", "python_identity_fp32_2"],
        ]
        
        for model_pair in model_pairs:
            
            threads = []
            for model_name in model_pair:
                threads.append(
                    threading.Thread(
                        target=triton_client.load_model, args=(model_name,)
                    )
                )
            start_time = time.time()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            end_time = time.time()
            loading_time = end_time - start_time
            
            
            
            self.assertLess(
                loading_time, 15.0, "Concurrent loading speedup not observed"
            )
            
            self.assertGreaterEqual(
                loading_time, 10.0, "Invalid concurrent loading time"
            )
            
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            for model_name in model_pair:
                self.assertTrue(triton_client.is_model_ready(model_name))

    def test_concurrent_model_load(self):
        
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        
        with concurrent.futures.ThreadPoolExecutor() as pool:
            
            thread_1 = pool.submit(triton_client.load_model, "identity_model")
            time.sleep(2)  
            
            shutil.move("models", "models_v1")
            shutil.move("models_v2", "models")
            
            thread_2 = pool.submit(triton_client.load_model, "identity_model")
            
            thread_1.result()
            thread_2.result()
        
        self.assertTrue(triton_client.is_server_live())
        self.assertTrue(triton_client.is_server_ready())
        self.assertTrue(triton_client.is_model_ready("identity_model"))
        
        model_metadata = triton_client.get_model_metadata("identity_model")
        self.assertEqual(model_metadata.platform, "python")

    def test_concurrent_model_load_unload(self):
        
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        
        
        with concurrent.futures.ThreadPoolExecutor() as pool:
            load_thread = pool.submit(triton_client.load_model, "identity_zero_1_int32")
            time.sleep(2)  
            unload_thread = pool.submit(
                triton_client.unload_model, "identity_zero_1_int32"
            )
            load_thread.result()
            unload_thread.result()
        self.assertTrue(triton_client.is_server_live())
        self.assertTrue(triton_client.is_server_ready())
        self.assertFalse(triton_client.is_model_ready("identity_zero_1_int32"))
        
        
        with concurrent.futures.ThreadPoolExecutor() as pool:
            load_thread = pool.submit(
                triton_client.load_model, "ensemble_zero_1_float32"
            )
            time.sleep(2)  
            unload_thread = pool.submit(
                triton_client.unload_model, "custom_zero_1_float32"
            )
            load_thread.result()
            unload_thread.result()
        self.assertTrue(triton_client.is_server_live())
        self.assertTrue(triton_client.is_server_ready())
        self.assertFalse(triton_client.is_model_ready("ensemble_zero_1_float32"))
        self.assertFalse(triton_client.is_model_ready("custom_zero_1_float32"))
        
        model_names = ["identity_zero_1_int32", "ensemble_zero_1_float32"]
        for is_load in [True, False]:
            action_fn = (
                triton_client.load_model if is_load else triton_client.unload_model
            )
            with concurrent.futures.ThreadPoolExecutor() as pool:
                threads = []
                for model_name in model_names:
                    threads.append(pool.submit(action_fn, model_name))
                for thread in concurrent.futures.as_completed(threads):
                    thread.result()
            for model_name in model_names:
                self.assertEqual(is_load, triton_client.is_model_ready(model_name))

    
    
    
    
    
    
    
    
    
    
    def test_concurrent_same_model_load_unload_stress(self):
        model_name = "identity_zero_1_int32"
        num_threads = 32
        num_iterations = 1024
        try:
            triton_client = grpcclient.InferenceServerClient(
                "localhost:8001", verbose=True
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        load_fail_reasons = [
            "unexpected miss in global map",
            "no version is available",
            "failed to poll from model repository",
        ]
        unload_fail_reasons = ["versions that are still available: 1"]
        load_fail_messages = [
            ("failed to load '" + model_name + "', " + reason)
            for reason in load_fail_reasons
        ]
        unload_fail_messages = [
            ("failed to unload '" + model_name + "', " + reason)
            for reason in unload_fail_reasons
        ]
        global_exception_stats = {}  
        load_before_unload_finish = [False]  

        def _load_unload():
            exception_stats = {}  
            for i in range(num_iterations):
                try:
                    triton_client.load_model(model_name)
                except InferenceServerException as ex:
                    
                    
                    error_message = ex.message()
                    self.assertIn(error_message, load_fail_messages)
                    if error_message not in exception_stats:
                        exception_stats[error_message] = 0
                    exception_stats[error_message] += 1
                try:
                    triton_client.unload_model(model_name)
                except InferenceServerException as ex:
                    
                    
                    error_message = ex.message()
                    self.assertIn(error_message, unload_fail_messages)
                    if error_message not in exception_stats:
                        exception_stats[error_message] = 0
                    exception_stats[error_message] += 1
                    load_before_unload_finish[0] = True
            return exception_stats

        with concurrent.futures.ThreadPoolExecutor() as pool:
            threads = []
            for i in range(num_threads):
                threads.append(pool.submit(_load_unload))
            for t in threads:
                exception_stats = t.result()
                for key, count in exception_stats.items():
                    if key not in global_exception_stats:
                        global_exception_stats[key] = 0
                    global_exception_stats[key] += count

        self.assertTrue(triton_client.is_server_live())
        self.assertTrue(triton_client.is_server_ready())

        
        
        
        if load_before_unload_finish[0] == False:
            
            warning_msg = "Cannot replicate a load while async unloading. CPU count: {}. num_threads: {}.".format(
                multiprocessing.cpu_count(), num_threads
            )
            global_exception_stats[warning_msg] = 1

        stats_path = "./test_concurrent_same_model_load_unload_stress.statistics.log"
        with open(stats_path, mode="w", encoding="utf-8") as f:
            f.write(str(global_exception_stats) + "\n")

    def test_concurrent_model_instance_load_speedup(self):
        
        try:
            triton_client = httpclient.InferenceServerClient(
                "localhost:8000", verbose=True
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        models = ["identity_fp32"]
        
        num_instances = 2
        instance_group = [{"kind": "KIND_CPU", "count": num_instances}]
        config = {"instance_group": instance_group}
        for model in models:
            
            start_time = time.time()
            try:
                triton_client.load_model(model, config=json.dumps(config))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))
            end_time = time.time()
            loading_time = end_time - start_time
            print(f"Time to load {num_instances} instances: {loading_time}")

            
            
            
            self.assertLess(
                loading_time, 15.0, "Concurrent loading speedup not observed"
            )
            
            self.assertGreaterEqual(
                loading_time, 10.0, "Invalid concurrent loading time"
            )
            
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model))

    def _call_with_timeout(self, callable, timeout_secs):
        
        def timeout_handler(sig, frame):
            raise TimeoutError()

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_secs)
        result = callable()
        return result

    def _call_with_expected_timeout(self, callable, timeout_secs=3):
        
        try:
            self._call_with_timeout(callable, timeout_secs)
        except TimeoutError:
            print("Inference timed out as expected.")
            return
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))
        else:
            self.assertTrue(False, "unexpected success, call should've timed out.")

    def _get_fp32_io(self, client_type):
        
        input_names = ["INPUT0", "INPUT1"]
        output_names = ["OUTPUT0", "OUTPUT1"]
        dtype, dims, shape = ("TYPE_FP32", [-1, 16], [1, 16])
        input_config = [
            {"name": name, "data_type": dtype, "dims": dims} for name in input_names
        ]
        output_config = [
            {"name": name, "data_type": dtype, "dims": dims} for name in output_names
        ]
        
        inputs = []
        for name in input_names:
            inputs.append(
                client_type.InferInput(name, shape, dtype.replace("TYPE_", ""))
            )
            inputs[-1].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        return input_config, output_config, inputs

    def test_concurrent_model_instance_load_sanity(self):
        cpu, gpu = "KIND_CPU", "KIND_GPU"
        default_kinds = [cpu, gpu]
        backend_kinds = {"plan": [gpu], "openvino": [cpu]}
        try:
            client_type = httpclient
            triton_client = client_type.InferenceServerClient(
                "localhost:8000", verbose=True
            )
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        backends = os.environ.get("PARALLEL_BACKENDS", "").split()
        self.assertTrue(len(backends) > 0, "PARALLEL_BACKENDS wasn't set")

        num_instances = 5
        input_config, output_config, inputs = self._get_fp32_io(client_type)
        for backend in backends:
            model = tu.get_model_name(backend, np.float32, np.float32, np.float32)
            kinds = backend_kinds.get(backend, default_kinds)
            for kind in kinds:
                with self.subTest(backend=backend, model=model, kind=kind):
                    
                    instance_group = {"kind": kind, "count": num_instances}
                    
                    
                    
                    max_batch_size = 0
                    sequence_timeout_secs = 10
                    sequence_batching = {
                        "direct": {},
                        "max_sequence_idle_microseconds": sequence_timeout_secs
                        * 1000000,
                    }
                    config = {
                        "instance_group": instance_group,
                        "max_batch_size": max_batch_size,
                        "sequence_batching": sequence_batching,
                        "input": input_config,
                        "output": output_config,
                    }
                    print(
                        f"~~~ Backend: [{backend}], Model: [{model}], Config: [{config}] ~~~"
                    )
                    
                    try:
                        triton_client.load_model(model, config=json.dumps(config))
                    except Exception as ex:
                        self.assertTrue(False, "unexpected error {}".format(ex))

                    
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_model_ready(model))
                    print(
                        "Model Repository Index after load:",
                        triton_client.get_model_repository_index(),
                    )

                    
                    for i in range(1, num_instances + 1):
                        try:
                            triton_client.infer(
                                model, inputs, sequence_id=i, sequence_start=True
                            )
                        except Exception as ex:
                            self.assertTrue(
                                False, "unexpected inference error {}".format(ex)
                            )

                    
                    
                    
                    callable = partial(
                        triton_client.infer,
                        model,
                        inputs,
                        sequence_id=num_instances + 1,
                        sequence_start=True,
                    )
                    self._call_with_expected_timeout(callable, timeout_secs=3)

                    
                    try:
                        triton_client.unload_model(model)
                    except Exception as ex:
                        self.assertTrue(False, "unexpected error {}".format(ex))

                    
                    num_tries = 10
                    for i in range(num_tries):
                        if triton_client.is_server_ready():
                            break
                        print(
                            f"[Attempt {i}] Server not ready yet, sleeping and retrying. Current repository index: {triton_client.get_model_repository_index()}"
                        )
                        time.sleep(6)
                    print(
                        "Model Repository Index after unload attempts:",
                        triton_client.get_model_repository_index(),
                    )
                    self.assertTrue(triton_client.is_server_ready())

    def test_model_config_overwite(self):
        model_name = "identity_fp32"

        
        try:
            triton_client = self._get_client()
            self.assertTrue(triton_client.is_server_live())
            self.assertTrue(triton_client.is_server_ready())
            self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        
        original_config = triton_client.get_model_config(model_name)

        
        
        
        override_config = 

        
        self.assertTrue(original_config != None and original_config != override_config)

        
        triton_client.load_model(model_name, config=override_config)

        
        updated_config = triton_client.get_model_config(model_name)

        
        triton_client.load_model(model_name)

        
        updated_config2 = triton_client.get_model_config(model_name)
        self.assertEqual(updated_config, updated_config2)

        
        
        time.sleep(0.1)  
        Path(os.path.join("models", model_name, "config.pbtxt")).touch()

        
        triton_client.load_model(model_name)

        
        updated_config = triton_client.get_model_config(model_name)
        self.assertEqual(original_config, updated_config)

    def test_shutdown_while_background_unloading(self):
        model_name = "identity_fp32"
        triton_client = self._get_client()
        self.assertTrue(triton_client.is_server_live())
        self.assertTrue(triton_client.is_server_ready())
        
        self.assertTrue(triton_client.is_model_ready(model_name, "1"))
        python_model_config = triton_client.get_model_config(model_name)
        self.assertEqual(python_model_config["backend"], "python")
        
        
        override_config = "{\n"
        override_config += '"name": "identity_fp32",\n'
        override_config += '"backend": "identity"\n'
        override_config += "}"
        triton_client.load_model(model_name, config=override_config)
        identity_model_config = triton_client.get_model_config(model_name)
        self.assertEqual(identity_model_config["backend"], "identity")
        
        

    def test_shutdown_while_loading(self):
        triton_client = self._get_client()
        self.assertTrue(triton_client.is_server_live())
        self.assertTrue(triton_client.is_server_ready())
        
        model_name = "identity_fp32"
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(triton_client.load_model, model_name)
        self.assertFalse(triton_client.is_model_ready(model_name))
        
        

    def test_shutdown_with_live_connection(self):
        model_name = "add_sub"
        model_shape = (16,)
        from geventhttpclient.response import HTTPConnectionClosed

        input_data = np.ones(shape=model_shape, dtype=np.float32)
        inputs = [
            httpclient.InferInput("INPUT0", model_shape, "FP32"),
            httpclient.InferInput("INPUT1", model_shape, "FP32"),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(input_data)

        
        conn = httpclient.InferenceServerClient("localhost:8000", verbose=True)
        conn.infer(model_name, inputs)

        
        os.kill(int(os.environ["SERVER_PID"]), signal.SIGINT)
        time.sleep(2)

        
        conn.infer(model_name, inputs)

        
        conn.close()
        time.sleep(3)

        
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertIn(
            "Waiting for in-flight requests to complete.",
            server_log,
            "precondition not met - core shutdown did not begin",
        )
        self.assertEqual(
            server_log.count("Timeout 30: "),
            1,
            "exit timeout countdown restart detected",
        )

    def test_add_custom_config(self):
        models_base = ("libtorch",)
        models = list()
        for m in models_base:
            models.append(tu.get_model_name(m, np.float32, np.float32, np.float32))

        
        for model_name in models:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        
        
        for base_name, model_name in zip(models_base, models):
            shutil.copyfile(
                "config.pbtxt.custom." + base_name,
                "models/" + model_name + "/configs/custom.pbtxt",
            )

        time.sleep(5)  
        for model_name in models:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "2"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_delete_custom_config(self):
        models_base = ("libtorch",)
        models = list()
        for m in models_base:
            models.append(tu.get_model_name(m, np.float32, np.float32, np.float32))

        
        for model_name in models:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertFalse(triton_client.is_model_ready(model_name, "1"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "2"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

        
        
        
        for model_name in models:
            os.remove("models/" + model_name + "/configs/custom.pbtxt")

        time.sleep(5)  
        for model_name in models:
            try:
                for triton_client in (
                    httpclient.InferenceServerClient("localhost:8000", verbose=True),
                    grpcclient.InferenceServerClient("localhost:8001", verbose=True),
                ):
                    self.assertTrue(triton_client.is_server_live())
                    self.assertTrue(triton_client.is_server_ready())
                    self.assertTrue(triton_client.is_model_ready(model_name, "1"))
                    self.assertFalse(triton_client.is_model_ready(model_name, "2"))
                    self.assertTrue(triton_client.is_model_ready(model_name, "3"))
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_load_new_model_version(self):
        model_name = "identity_fp32"
        client = self._get_client(use_grpc=True)

        
        
        
        self.assertTrue(client.is_model_ready(model_name, "1"))
        self.assertTrue(client.is_model_ready(model_name, "2"))
        self.assertFalse(client.is_model_ready(model_name, "3"))
        self.assertFalse(client.is_model_ready(model_name, "4"))
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertEqual(server_log.count("[PB model] Loading version 1"), 1)
        self.assertEqual(server_log.count("[PB model] Loading version 2"), 1)
        self.assertEqual(server_log.count("[PB model] Loading version 3"), 0)
        self.assertEqual(server_log.count("[PB model] Loading version 4"), 0)
        self.assertEqual(server_log.count("successfully loaded 'identity_fp32'"), 1)

        
        Path(os.path.join("models", model_name, "2", "model.py")).touch()
        
        src_path = os.path.join("models", model_name, "3")
        dst_path = os.path.join("models", model_name, "4")
        shutil.copytree(src_path, dst_path)
        
        config_path = os.path.join("models", model_name, "config.pbtxt")
        with open(config_path, mode="r+", encoding="utf-8", errors="strict") as f:
            config = f.read()
            config = config.replace(
                "version_policy: { specific: { versions: [1, 2] } }",
                "version_policy: { specific: { versions: [1, 2, 3, 4] } }",
            )
            f.truncate(0)
            f.seek(0)
            f.write(config)
        
        time.sleep(0.1)
        
        client.load_model(model_name)

        
        
        
        
        self.assertTrue(client.is_model_ready(model_name, "1"))
        self.assertTrue(client.is_model_ready(model_name, "2"))
        self.assertTrue(client.is_model_ready(model_name, "3"))
        self.assertTrue(client.is_model_ready(model_name, "4"))
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertEqual(server_log.count("[PB model] Loading version 1"), 1)
        self.assertEqual(server_log.count("[PB model] Loading version 2"), 2)
        self.assertEqual(server_log.count("[PB model] Loading version 3"), 1)
        self.assertEqual(server_log.count("[PB model] Loading version 4"), 1)
        self.assertEqual(server_log.count("successfully loaded 'identity_fp32'"), 2)

        
        Path(os.path.join("models", model_name, "dummy_dependency.py")).touch()
        
        time.sleep(0.1)
        
        client.load_model(model_name)

        
        self.assertTrue(client.is_model_ready(model_name, "1"))
        self.assertTrue(client.is_model_ready(model_name, "2"))
        self.assertTrue(client.is_model_ready(model_name, "3"))
        self.assertTrue(client.is_model_ready(model_name, "4"))
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertEqual(server_log.count("[PB model] Loading version 1"), 2)
        self.assertEqual(server_log.count("[PB model] Loading version 2"), 3)
        self.assertEqual(server_log.count("[PB model] Loading version 3"), 2)
        self.assertEqual(server_log.count("[PB model] Loading version 4"), 2)
        self.assertEqual(server_log.count("successfully loaded 'identity_fp32'"), 3)

        
        config_path = os.path.join("models", model_name, "config.pbtxt")
        with open(config_path, mode="r+", encoding="utf-8", errors="strict") as f:
            config = f.read()
            config = config.replace(
                "version_policy: { specific: { versions: [1, 2, 3, 4] } }",
                "version_policy: { specific: { versions: [4] } }",
            )
            f.truncate(0)
            f.seek(0)
            f.write(config)
        
        time.sleep(0.1)
        
        client.load_model(model_name)

        
        self.assertFalse(client.is_model_ready(model_name, "1"))
        self.assertFalse(client.is_model_ready(model_name, "2"))
        self.assertFalse(client.is_model_ready(model_name, "3"))
        self.assertTrue(client.is_model_ready(model_name, "4"))
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertEqual(server_log.count("[PB model] Loading version 1"), 2)
        self.assertEqual(server_log.count("[PB model] Loading version 2"), 3)
        self.assertEqual(server_log.count("[PB model] Loading version 3"), 2)
        self.assertEqual(server_log.count("[PB model] Loading version 4"), 2)
        self.assertEqual(server_log.count("successfully loaded 'identity_fp32'"), 4)

        
        config_path = os.path.join("models", model_name, "config.pbtxt")
        with open(config_path, mode="r+", encoding="utf-8", errors="strict") as f:
            config = f.read()
            config = config.replace(
                "version_policy: { specific: { versions: [4] } }",
                "version_policy: { specific: { versions: [1, 4] } }",
            )
            f.truncate(0)
            f.seek(0)
            f.write(config)
        
        time.sleep(0.1)
        
        client.load_model(model_name)

        
        self.assertTrue(client.is_model_ready(model_name, "1"))
        self.assertFalse(client.is_model_ready(model_name, "2"))
        self.assertFalse(client.is_model_ready(model_name, "3"))
        self.assertTrue(client.is_model_ready(model_name, "4"))
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertEqual(server_log.count("[PB model] Loading version 1"), 3)
        self.assertEqual(server_log.count("[PB model] Loading version 2"), 3)
        self.assertEqual(server_log.count("[PB model] Loading version 3"), 2)
        self.assertEqual(server_log.count("[PB model] Loading version 4"), 2)
        self.assertEqual(server_log.count("successfully loaded 'identity_fp32'"), 5)


if __name__ == "__main__":
    unittest.main()
