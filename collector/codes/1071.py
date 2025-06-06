import os
import sys

sys.path.append(os.path.join(os.environ["TRITON_QA_ROOT_DIR"], "common"))

import shutil
import time
import unittest

import numpy as np
import test_util as tu
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class AddSubChecker:

    def __init__(self, checker_client=None):

        if checker_client is None:
            import tritonclient.http as checker_client
        if "http" in checker_client.__name__:
            self.client_ = checker_client.InferenceServerClient("localhost:8000")
        else:
            self.client_ = checker_client.InferenceServerClient("localhost:8001")

        self.inputs_ = []
        self.inputs_.append(checker_client.InferInput("INPUT0", [16], "INT32"))
        self.inputs_.append(checker_client.InferInput("INPUT1", [16], "INT32"))

        input_data = np.arange(start=0, stop=16, dtype=np.int32)
        self.inputs_[0].set_data_from_numpy(input_data)
        self.inputs_[1].set_data_from_numpy(input_data)
        self.expected_outputs_ = {
            "add": (input_data + input_data),
            "sub": (input_data - input_data),
        }

    def infer(self, model):
        res = self.client_.infer(model, self.inputs_)
        np.testing.assert_allclose(
            res.as_numpy("OUTPUT0"), self.expected_outputs_["add"]
        )
        np.testing.assert_allclose(
            res.as_numpy("OUTPUT1"), self.expected_outputs_["sub"]
        )


class SubAddChecker(AddSubChecker):
    def infer(self, model):
        res = self.client_.infer(model, self.inputs_)
        np.testing.assert_allclose(
            res.as_numpy("OUTPUT0"), self.expected_outputs_["sub"]
        )
        np.testing.assert_allclose(
            res.as_numpy("OUTPUT1"), self.expected_outputs_["add"]
        )


class ModelNamespacePoll(tu.TestResultCollector):
    def setUp(self):
        self.addsub_ = AddSubChecker()
        self.subadd_ = SubAddChecker()

        self.client_ = httpclient.InferenceServerClient("localhost:8000")

    def check_health(self, expect_live=True, expect_ready=True):
        self.assertEqual(self.client_.is_server_live(), expect_live)
        self.assertEqual(self.client_.is_server_ready(), expect_ready)

    def test_no_duplication(self):

        self.check_health()

        for model in ["simple_addsub", "composing_addsub"]:
            self.addsub_.infer(model)
        for model in ["simple_subadd", "composing_subadd"]:
            self.subadd_.infer(model)

    def test_duplication(self):

        self.check_health()

        for model in [
            "simple_addsub",
        ]:
            self.addsub_.infer(model)
        for model in [
            "simple_subadd",
        ]:
            self.subadd_.infer(model)

        try:
            self.addsub_.infer("composing_model")
            self.assertTrue(False, "expected error for inferring ambiguous named model")
        except InferenceServerException as ex:
            self.assertIn("ambiguity", ex.message())

    def test_ensemble_duplication(self):

        self.check_health()

        for model in [
            "composing_addsub",
        ]:
            self.addsub_.infer(model)
        for model in [
            "composing_subadd",
        ]:
            self.subadd_.infer(model)

        try:
            self.addsub_.infer("simple_ensemble")
            self.assertTrue(False, "expected error for inferring ambiguous named model")
        except InferenceServerException as ex:
            self.assertIn("ambiguity", ex.message())

    def test_dynamic_resolution(self):

        self.assertTrue("NAMESPACE_TESTING_DIRCTORY" in os.environ)
        td = os.environ["NAMESPACE_TESTING_DIRCTORY"]
        composing_before_path = os.path.join(td, "addsub_repo", "composing_model")
        composing_after_path = os.path.join(td, "composing_model")

        self.check_health()

        shutil.move(composing_before_path, composing_after_path)
        time.sleep(5)

        for model in ["simple_subadd", "simple_addsub", "composing_model"]:
            self.subadd_.infer(model)

        shutil.move(composing_after_path, composing_before_path)
        time.sleep(5)

        for model in [
            "simple_addsub",
        ]:
            self.addsub_.infer(model)
        for model in [
            "simple_subadd",
        ]:
            self.subadd_.infer(model)

        try:
            self.addsub_.infer("composing_model")
            self.assertTrue(False, "expected error for inferring ambiguous named model")
        except InferenceServerException as ex:
            self.assertIn("ambiguity", ex.message())


class ModelNamespaceExplicit(tu.TestResultCollector):
    def setUp(self):
        self.addsub_ = AddSubChecker()
        self.subadd_ = SubAddChecker()

        self.client_ = httpclient.InferenceServerClient("localhost:8000")

    def check_health(self, expect_live=True, expect_ready=True):
        self.assertEqual(self.client_.is_server_live(), expect_live)
        self.assertEqual(self.client_.is_server_ready(), expect_ready)

    def test_no_duplication(self):

        self.check_health()

        for model in ["simple_addsub", "simple_subadd"]:
            self.client_.load_model(model)

        for model in ["simple_addsub", "composing_addsub"]:
            self.addsub_.infer(model)
        for model in ["simple_subadd", "composing_subadd"]:
            self.subadd_.infer(model)

    def test_duplication(self):

        self.check_health()

        for model in ["simple_addsub", "simple_subadd"]:
            self.client_.load_model(model)

        for model in [
            "simple_addsub",
        ]:
            self.addsub_.infer(model)
        for model in [
            "simple_subadd",
        ]:
            self.subadd_.infer(model)

        try:
            self.addsub_.infer("composing_model")
            self.assertTrue(False, "expected error for inferring ambiguous named model")
        except InferenceServerException as ex:
            self.assertIn("ambiguity", ex.message())

    def test_ensemble_duplication(self):

        self.check_health()

        for model in ["simple_ensemble"]:
            self.client_.load_model(model)

        for model in [
            "composing_addsub",
        ]:
            self.addsub_.infer(model)
        for model in [
            "composing_subadd",
        ]:
            self.subadd_.infer(model)

        try:
            self.addsub_.infer("simple_ensemble")
            self.assertTrue(False, "expected error for inferring ambiguous named model")
        except InferenceServerException as ex:
            self.assertIn("ambiguity", ex.message())

    def test_dynamic_resolution(self):

        self.assertTrue("NAMESPACE_TESTING_DIRCTORY" in os.environ)
        td = os.environ["NAMESPACE_TESTING_DIRCTORY"]
        composing_before_path = os.path.join(td, "addsub_repo", "composing_model")
        composing_after_path = os.path.join(td, "composing_model")

        self.check_health()

        shutil.move(composing_before_path, composing_after_path)

        for model in ["simple_addsub", "simple_subadd"]:
            self.client_.load_model(model)

        for model in ["simple_subadd", "simple_addsub", "composing_model"]:
            self.subadd_.infer(model)

        shutil.move(composing_after_path, composing_before_path)

        for model in [
            "simple_addsub",
        ]:
            self.client_.load_model(model)

        for model in [
            "simple_addsub",
        ]:
            self.addsub_.infer(model)
        for model in [
            "simple_subadd",
        ]:
            self.subadd_.infer(model)

        try:
            self.addsub_.infer("composing_model")
            self.assertTrue(False, "expected error for inferring ambiguous named model")
        except InferenceServerException as ex:
            self.assertIn("ambiguity", ex.message())


if __name__ == "__main__":
    unittest.main()
