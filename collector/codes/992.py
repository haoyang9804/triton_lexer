

























import json
import sys
import threading
import time

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


class TritonPythonModel:
    

    def initialize(self, args):
        logger = pb_utils.Logger
        logger.log("Initialize-Specific Msg!", logger.INFO)
        logger.log_info("Initialize-Info Msg!")
        logger.log_warn("Initialize-Warning Msg!")
        logger.log_error("Initialize-Error Msg!")
        
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                .format(
                    args["model_name"]
                )
            )

        
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        
        self.out_dtype = pb_utils.triton_string_to_numpy(out_config["data_type"])

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()
        logger = pb_utils.Logger
        logger.log("Initialize-Specific Msg!", logger.INFO)
        logger.log_info("Initialize-Info Msg!")
        logger.log_warn("Initialize-Warning Msg!")
        logger.log_error("Initialize-Error Msg!")

    def execute(self, requests):
        
        logger = pb_utils.Logger
        logger.log("Execute-Specific Msg!", logger.INFO)
        logger.log_info("Execute-Info Msg!")
        logger.log_warn("Execute-Warning Msg!")
        logger.log_error("Execute-Error Msg!")
        
        for i, request in enumerate(requests):
            request_input = pb_utils.get_input_tensor_by_name(request, "IN")

            
            infer_request = pb_utils.InferenceRequest(
                model_name="identity_fp32",
                requested_output_names=["OUTPUT0"],
                inputs=[pb_utils.Tensor("INPUT0", request_input.as_numpy())],
            )
            infer_response = infer_request.exec()
            if infer_response.has_error():
                raise pb_utils.TritonModelException(
                    f"BLS Response has an error: {infer_response.error().message()}"
                )

            output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
            if np.any(output0.as_numpy() != request_input.as_numpy()):
                raise pb_utils.TritonModelException(
                    f"BLS Request input and BLS response output do not match. {request_input.as_numpy()} != {output0.as_numpy()}"
                )

            thread1 = threading.Thread(
                target=self.response_thread,
                args=(
                    request.get_response_sender(),
                    pb_utils.get_input_tensor_by_name(request, "IN").as_numpy(),
                ),
            )
            thread1.daemon = True
            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1
            thread1.start()

        logger = pb_utils.Logger
        logger.log("Execute-Specific Msg!", logger.INFO)
        logger.log_info("Execute-Info Msg!")
        logger.log_warn("Execute-Warning Msg!")
        logger.log_error("Execute-Error Msg!")

        return None

    def _get_gpu_bls_outputs(self, input0_pb, input1_pb):
        
        logger = pb_utils.Logger
        logger.log("_get_gpu_bls_outputs-Specific Msg!", logger.INFO)
        logger.log_info("_get_gpu_bls_outputs-Info Msg!")
        logger.log_warn("_get_gpu_bls_outputs-Warning Msg!")
        logger.log_error("_get_gpu_bls_outputs-Error Msg!")

        infer_request = pb_utils.InferenceRequest(
            model_name="dlpack_add_sub",
            inputs=[input0_pb, input1_pb],
            requested_output_names=["OUTPUT0", "OUTPUT1"],
        )
        infer_response = infer_request.exec()
        if infer_response.has_error():
            return False

        output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
        output1 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT1")
        if output0 is None or output1 is None:
            return False

        
        
        if not input0_pb.is_cpu() or not input1_pb.is_cpu():
            if output0.is_cpu() or output1.is_cpu():
                return False
        else:
            if (not output0.is_cpu()) or (not output1.is_cpu()):
                return False

        
        
        rc_before_dlpack_output0 = sys.getrefcount(output0)
        rc_before_dlpack_output1 = sys.getrefcount(output1)

        output0_dlpack = output0.to_dlpack()
        output1_dlpack = output1.to_dlpack()

        rc_after_dlpack_output0 = sys.getrefcount(output0)
        rc_after_dlpack_output1 = sys.getrefcount(output1)

        if rc_after_dlpack_output0 - rc_before_dlpack_output0 != 1:
            return False

        if rc_after_dlpack_output1 - rc_before_dlpack_output1 != 1:
            return False

        
        output0_dlpack = None
        output1_dlpack = None
        rc_after_del_dlpack_output0 = sys.getrefcount(output0)
        rc_after_del_dlpack_output1 = sys.getrefcount(output1)
        if rc_after_del_dlpack_output0 - rc_after_dlpack_output0 != -1:
            return False

        if rc_after_del_dlpack_output1 - rc_after_dlpack_output1 != -1:
            return False

        return output0.to_dlpack(), output1.to_dlpack()

    def _test_gpu_bls_add_sub(self, is_input0_gpu, is_input1_gpu):
        logger = pb_utils.Logger
        logger.log("_test_gpu_bls_add_sub-Specific Msg!", logger.INFO)
        logger.log_info("_test_gpu_bls_add_sub-Info Msg!")
        logger.log_warn("_test_gpu_bls_add_sub-Warning Msg!")
        logger.log_error("_test_gpu_bls_add_sub-Error Msg!")

        input0 = torch.rand(16)
        input1 = torch.rand(16)

        if is_input0_gpu:
            input0 = input0.to("cuda")

        if is_input1_gpu:
            input1 = input1.to("cuda")

        input0_pb = pb_utils.Tensor.from_dlpack("INPUT0", to_dlpack(input0))
        input1_pb = pb_utils.Tensor.from_dlpack("INPUT1", to_dlpack(input1))
        gpu_bls_return = self._get_gpu_bls_outputs(input0_pb, input1_pb)
        if gpu_bls_return:
            output0_dlpack, output1_dlpack = gpu_bls_return
        else:
            return False

        expected_output_0 = from_dlpack(input0_pb.to_dlpack()).to("cpu") + from_dlpack(
            input1_pb.to_dlpack()
        ).to("cpu")
        expected_output_1 = from_dlpack(input0_pb.to_dlpack()).to("cpu") - from_dlpack(
            input1_pb.to_dlpack()
        ).to("cpu")

        output0_matches = torch.all(
            expected_output_0 == from_dlpack(output0_dlpack).to("cpu")
        )
        output1_matches = torch.all(
            expected_output_1 == from_dlpack(output1_dlpack).to("cpu")
        )
        if not output0_matches or not output1_matches:
            return False

        return True

    def execute_gpu_bls(self):
        logger = pb_utils.Logger
        logger.log("execute_gpu_bls-Specific Msg!", logger.INFO)
        logger.log_info("execute_gpu_bls-Info Msg!")
        logger.log_warn("execute_gpu_bls-Warning Msg!")
        logger.log_error("execute_gpu_bls-Error Msg!")
        for input0_device in [True, False]:
            for input1_device in [True, False]:
                test_status = self._test_gpu_bls_add_sub(input0_device, input1_device)
                if not test_status:
                    return False

        return True

    def response_thread(self, response_sender, in_input):
        
        
        
        logger = pb_utils.Logger
        logger.log("response_thread-Specific Msg!", logger.INFO)
        logger.log_info("response_thread-Info Msg!")
        logger.log_warn("response_thread-Warning Msg!")
        logger.log_error("response_thread-Error Msg!")
        time.sleep(5)

        
        if sys.platform != "win32":
            status = self.execute_gpu_bls()
        else:
            status = True

        if not status:
            infer_response = pb_utils.InferenceResponse(error="GPU BLS test failed.")
            response_sender.send(infer_response)
        else:
            in_value = in_input
            infer_request = pb_utils.InferenceRequest(
                model_name="identity_fp32",
                requested_output_names=["OUTPUT0"],
                inputs=[pb_utils.Tensor("INPUT0", in_input)],
            )
            infer_response = infer_request.exec()
            output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
            if infer_response.has_error():
                response = pb_utils.InferenceResponse(
                    error=infer_response.error().message()
                )
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
            elif np.any(in_input != output0.as_numpy()):
                error_message = (
                    "BLS Request input and BLS response output do not match."
                    f" {in_value} != {output0.as_numpy()}"
                )
                response = pb_utils.InferenceResponse(error=error_message)
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
            else:
                output_tensors = [pb_utils.Tensor("OUT", in_value)]
                response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1
        logger.log("response_thread-Specific Msg!", logger.INFO)
        logger.log_info("response_thread-Info Msg!")
        logger.log_warn("response_thread-Warning Msg!")
        logger.log_error("response_thread-Error Msg!")

    def finalize(self):
        
        logger = pb_utils.Logger
        logger.log_info("Finalize invoked")

        inflight_threads = True
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
            if inflight_threads:
                time.sleep(0.1)

        logger.log_info("Finalize complete...")
