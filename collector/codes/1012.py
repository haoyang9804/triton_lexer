import os
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class ResponseSenderTest(unittest.TestCase):
    _inputs_parameters_zero_response_pre_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": True,
        "return_a_response": False,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_zero_response_post_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": False,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": True,
    }
    _inputs_parameters_one_response_pre_return = {
        "number_of_response_before_return": 1,
        "send_complete_final_flag_before_return": True,
        "return_a_response": False,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_one_response_post_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": False,
        "number_of_response_after_return": 1,
        "send_complete_final_flag_after_return": True,
    }
    _inputs_parameters_two_response_pre_return = {
        "number_of_response_before_return": 2,
        "send_complete_final_flag_before_return": True,
        "return_a_response": False,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_two_response_post_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": False,
        "number_of_response_after_return": 2,
        "send_complete_final_flag_after_return": True,
    }
    _inputs_parameters_response_pre_and_post_return = {
        "number_of_response_before_return": 1,
        "send_complete_final_flag_before_return": False,
        "return_a_response": False,
        "number_of_response_after_return": 3,
        "send_complete_final_flag_after_return": True,
    }
    _inputs_parameters_one_response_on_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": True,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_one_response_pre_and_on_return = {
        "number_of_response_before_return": 1,
        "send_complete_final_flag_before_return": True,
        "return_a_response": True,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_one_response_on_and_post_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": True,
        "number_of_response_after_return": 1,
        "send_complete_final_flag_after_return": True,
    }

    def _get_inputs(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        shape = [1, 1]
        inputs = [
            grpcclient.InferInput("NUMBER_OF_RESPONSE_BEFORE_RETURN", shape, "UINT8"),
            grpcclient.InferInput(
                "SEND_COMPLETE_FINAL_FLAG_BEFORE_RETURN", shape, "BOOL"
            ),
            grpcclient.InferInput("RETURN_A_RESPONSE", shape, "BOOL"),
            grpcclient.InferInput("NUMBER_OF_RESPONSE_AFTER_RETURN", shape, "UINT8"),
            grpcclient.InferInput(
                "SEND_COMPLETE_FINAL_FLAG_AFTER_RETURN", shape, "BOOL"
            ),
        ]
        inputs[0].set_data_from_numpy(
            np.array([[number_of_response_before_return]], np.uint8)
        )
        inputs[1].set_data_from_numpy(
            np.array([[send_complete_final_flag_before_return]], bool)
        )
        inputs[2].set_data_from_numpy(np.array([[return_a_response]], bool))
        inputs[3].set_data_from_numpy(
            np.array([[number_of_response_after_return]], np.uint8)
        )
        inputs[4].set_data_from_numpy(
            np.array([[send_complete_final_flag_after_return]], bool)
        )
        return inputs

    def _generate_streaming_callback_and_responses_pair(self):
        responses = []

        def callback(result, error):
            responses.append({"result": result, "error": error})

        return callback, responses

    def _infer_parallel(self, model_name, parallel_inputs):
        callback, responses = self._generate_streaming_callback_and_responses_pair()
        with grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001") as client:
            client.start_stream(callback)
            for inputs in parallel_inputs:
                client.async_stream_infer(model_name, inputs)
            client.stop_stream()
        return responses

    def _infer(
        self,
        model_name,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        inputs = self._get_inputs(
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        return self._infer_parallel(model_name, [inputs])

    def _assert_responses_valid(
        self,
        responses,
        number_of_response_before_return,
        return_a_response,
        number_of_response_after_return,
    ):
        before_return_response_count = 0
        response_returned = False
        after_return_response_count = 0
        for response in responses:
            result, error = response["result"], response["error"]
            self.assertIsNone(error)
            result_np = result.as_numpy(name="INDEX")
            response_id = result_np.sum() / result_np.shape[0]
            if response_id < 1000:
                self.assertFalse(
                    response_returned,
                    "Expect at most one response returned per request.",
                )
                response_returned = True
            elif response_id < 2000:
                before_return_response_count += 1
            elif response_id < 3000:
                after_return_response_count += 1
            else:
                raise ValueError(f"Unexpected response_id: {response_id}")
        self.assertEqual(number_of_response_before_return, before_return_response_count)
        self.assertEqual(return_a_response, response_returned)
        self.assertEqual(number_of_response_after_return, after_return_response_count)

    def _assert_responses_exception(self, responses, expected_message):
        for response in responses:
            self.assertIsNone(response["result"])
            self.assertIsInstance(response["error"], InferenceServerException)
            self.assertIn(expected_message, response["error"].message())

        self.assertEqual(len(responses), 1)

    def _assert_decoupled_infer_success(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        model_name = "response_sender_decoupled"
        responses = self._infer(
            model_name,
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        self._assert_responses_valid(
            responses,
            number_of_response_before_return,
            return_a_response,
            number_of_response_after_return,
        )

        model_name = "response_sender_decoupled_async"
        responses = self._infer(
            model_name,
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        self._assert_responses_valid(
            responses,
            number_of_response_before_return,
            return_a_response,
            number_of_response_after_return,
        )

    def _assert_non_decoupled_infer_with_expected_response_success(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
        expected_number_of_response_before_return,
        expected_return_a_response,
        expected_number_of_response_after_return,
    ):
        model_name = "response_sender"
        responses = self._infer(
            model_name,
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

        model_name = "response_sender_async"
        responses = self._infer(
            model_name,
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

    def _assert_non_decoupled_infer_success(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        self._assert_non_decoupled_infer_with_expected_response_success(
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
            expected_number_of_response_before_return=number_of_response_before_return,
            expected_return_a_response=return_a_response,
            expected_number_of_response_after_return=number_of_response_after_return,
        )

    def test_decoupled_zero_response_pre_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_zero_response_pre_return
        )

    def test_decoupled_zero_response_post_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_zero_response_post_return
        )

    def test_decoupled_one_response_pre_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_one_response_pre_return
        )

    def test_decoupled_one_response_post_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_one_response_post_return
        )

    def test_decoupled_two_response_pre_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_two_response_pre_return
        )

    def test_decoupled_two_response_post_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_two_response_post_return
        )

    def test_decoupled_response_pre_and_post_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_response_pre_and_post_return
        )

    def test_non_decoupled_one_response_on_return(self):
        self._assert_non_decoupled_infer_success(
            **self._inputs_parameters_one_response_on_return
        )

    def test_non_decoupled_one_response_pre_return(self):
        self._assert_non_decoupled_infer_success(
            **self._inputs_parameters_one_response_pre_return
        )

    def test_non_decoupled_one_response_post_return(self):
        self._assert_non_decoupled_infer_success(
            **self._inputs_parameters_one_response_post_return
        )

    def test_decoupled_multiple_requests(self):
        parallel_inputs = [
            self._get_inputs(**self._inputs_parameters_zero_response_pre_return),
            self._get_inputs(**self._inputs_parameters_zero_response_post_return),
            self._get_inputs(**self._inputs_parameters_one_response_pre_return),
            self._get_inputs(**self._inputs_parameters_one_response_post_return),
            self._get_inputs(**self._inputs_parameters_two_response_pre_return),
            self._get_inputs(**self._inputs_parameters_two_response_post_return),
            self._get_inputs(**self._inputs_parameters_response_pre_and_post_return),
        ]
        expected_number_of_response_before_return = 4
        expected_return_a_response = False
        expected_number_of_response_after_return = 6

        model_name = "response_sender_decoupled_batching"
        responses = self._infer_parallel(model_name, parallel_inputs)
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

        model_name = "response_sender_decoupled_async_batching"
        responses = self._infer_parallel(model_name, parallel_inputs)
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

    def test_non_decoupled_multiple_requests(self):
        parallel_inputs = [
            self._get_inputs(**self._inputs_parameters_one_response_on_return),
            self._get_inputs(**self._inputs_parameters_one_response_pre_return),
            self._get_inputs(**self._inputs_parameters_one_response_post_return),
        ]
        expected_number_of_response_before_return = 1
        expected_return_a_response = True
        expected_number_of_response_after_return = 1

        model_name = "response_sender_batching"
        responses = self._infer_parallel(model_name, parallel_inputs)
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

        model_name = "response_sender_async_batching"
        responses = self._infer_parallel(model_name, parallel_inputs)
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

    def test_decoupled_one_response_on_return(self):
        responses = self._infer(
            model_name="response_sender_decoupled",
            **self._inputs_parameters_one_response_on_return,
        )
        self._assert_responses_exception(
            responses,
            expected_message="using the decoupled mode and the execute function must return None",
        )

    def test_decoupled_one_response_pre_and_on_return(self):

        responses = self._infer(
            model_name="response_sender_decoupled",
            **self._inputs_parameters_one_response_pre_and_on_return,
        )
        self._assert_responses_valid(
            responses,
            number_of_response_before_return=1,
            return_a_response=0,
            number_of_response_after_return=0,
        )

    def test_decoupled_one_response_on_and_post_return(self):

        responses = self._infer(
            model_name="response_sender_decoupled",
            **self._inputs_parameters_one_response_on_and_post_return,
        )
        self._assert_responses_exception(
            responses,
            expected_message="using the decoupled mode and the execute function must return None",
        )

    def test_non_decoupled_zero_response_pre_return(self):

        expected_message = (
            "Non-decoupled model cannot send complete final before sending a response"
        )
        model_name = "response_sender"
        responses = self._infer(
            model_name,
            **self._inputs_parameters_zero_response_pre_return,
        )
        self._assert_responses_exception(responses, expected_message)

        model_name = "response_sender_async"
        responses = self._infer(
            model_name,
            **self._inputs_parameters_zero_response_pre_return,
        )
        self._assert_responses_exception(responses, expected_message)

    @unittest.skip("Model unload will hang, see the TODO comment.")
    def test_non_decoupled_zero_response_post_return(self):

        raise NotImplementedError("No testing is performed")

    def test_non_decoupled_two_response_pre_return(self):

        self._assert_non_decoupled_infer_with_expected_response_success(
            **self._inputs_parameters_two_response_pre_return,
            expected_number_of_response_before_return=1,
            expected_return_a_response=False,
            expected_number_of_response_after_return=0,
        )

    @unittest.skip("Model unload will hang, see the TODO comment.")
    def test_non_decoupled_two_response_post_return(self):

        self._assert_non_decoupled_infer_with_expected_response_success(
            **self._inputs_parameters_two_response_post_return,
            expected_number_of_response_before_return=0,
            expected_return_a_response=False,
            expected_number_of_response_after_return=1,
        )

    def test_non_decoupled_one_response_pre_and_on_return(self):

        self._assert_non_decoupled_infer_with_expected_response_success(
            **self._inputs_parameters_one_response_pre_and_on_return,
            expected_number_of_response_before_return=1,
            expected_return_a_response=False,
            expected_number_of_response_after_return=0,
        )

    def test_non_decoupled_one_response_on_and_post_return(self):

        self._assert_non_decoupled_infer_with_expected_response_success(
            **self._inputs_parameters_one_response_on_and_post_return,
            expected_number_of_response_before_return=0,
            expected_return_a_response=True,
            expected_number_of_response_after_return=0,
        )


if __name__ == "__main__":
    unittest.main()
