

























import json
import threading
import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    

    def initialize(self, args):
        
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

    def execute(self, requests):
        

        
        for i, request in enumerate(requests):
            
            thread = threading.Thread(
                target=self.response_thread,
                args=(
                    request.get_response_sender(),
                    i,
                    pb_utils.get_input_tensor_by_name(request, "IN").as_numpy(),
                ),
            )
            thread.daemon = True

            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1

            thread.start()

        return None

    def response_thread(self, response_sender, index, in_input):
        
        
        
        

        in_value = in_input
        out_output = pb_utils.Tensor("OUT", in_value)

        if index == 0:
            error = pb_utils.TritonError("An error occurred during execution")
            response = pb_utils.InferenceResponse(
                output_tensors=[out_output], error=error
            )
        else:
            response = pb_utils.InferenceResponse(output_tensors=[out_output])
        response_sender.send(response)

        
        
        
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        
        print("Finalize invoked")

        inflight_threads = True
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
            if inflight_threads:
                time.sleep(0.1)

        print("Finalize complete...")
