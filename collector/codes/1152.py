

























import json
import threading

import numpy as np
import torch





import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

numpy_to_pytorch_dtype = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(model_config, "OUT")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                .format(
                    args["model_name"]
                )
            )

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        for request in requests:
            self.process_request(request)

        return None

    def process_request(self, request):
        
        
        thread = threading.Thread(
            target=self.response_thread,
            args=(
                request.get_response_sender(),
                pb_utils.get_input_tensor_by_name(request, "IN"),
                self.output_dtype,
            ),
        )

        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

    def response_thread(self, response_sender, in_input, output_dtype):
        
        

        for idx in range(in_input.as_numpy()[0]):
            if in_input.is_cpu():
                if (
                    in_input.as_numpy().dtype.type is np.bytes_
                    or in_input.as_numpy().dtype == np.object_
                ):
                    out_0 = in_input.as_numpy().astype(np.int32)
                    out_tensor = pb_utils.Tensor("OUT", out_0.astype(output_dtype))
                else:
                    in_0_pytorch = from_dlpack(in_input.to_dlpack())
                    out_0 = in_0_pytorch
                    if output_dtype == np.object_:
                        out_tensor = pb_utils.Tensor(
                            "OUT", out_0.numpy().astype(output_dtype)
                        )
                    else:
                        out_0 = out_0.type(numpy_to_pytorch_dtype[output_dtype])
                        out_tensor = pb_utils.Tensor.from_dlpack(
                            "OUT", to_dlpack(out_0)
                        )
            else:
                in_0_pytorch = from_dlpack(in_input.to_dlpack()).cuda()
                out_0 = in_0_pytorch
                out_tensor = pb_utils.Tensor.from_dlpack("OUTPUT0", to_dlpack(out_0))

            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            response_sender.send(response)

        
        
        
        
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1
