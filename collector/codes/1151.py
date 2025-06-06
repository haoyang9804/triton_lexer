import threading
import time

import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


class TritonPythonModel:

    def initialize(self, args):
        self._model_name = args["model_name"]
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def response_thread(self, response_sender, input0, gpu_output):

        time.sleep(5)

        if input0.is_cpu():
            if not gpu_output[0]:
                output0 = pb_utils.Tensor.from_dlpack("OUTPUT0", input0.to_dlpack())
            else:
                outptu0_pytorch = from_dlpack(input0.to_dlpack()).cuda()
                output0 = pb_utils.Tensor.from_dlpack(
                    "OUTPUT0", to_dlpack(outptu0_pytorch)
                )
        else:
            if gpu_output[0]:
                output0 = pb_utils.Tensor.from_dlpack("OUTPUT0", input0.to_dlpack())
            else:
                output0_pytorch = from_dlpack(input0.to_dlpack()).cpu()
                output0 = pb_utils.Tensor.from_dlpack(
                    "OUTPUT0", to_dlpack(output0_pytorch)
                )

        next_gpu_output = pb_utils.Tensor("NEXT_GPU_OUTPUT", gpu_output[1:])
        infer_response = pb_utils.InferenceResponse([output0, next_gpu_output])

        response_repeat = 2
        for _ in range(response_repeat):
            response_sender.send(infer_response)

        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def execute(self, requests):
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            gpu_output = pb_utils.get_input_tensor_by_name(
                request, "GPU_OUTPUT"
            ).as_numpy()

            thread = threading.Thread(
                target=self.response_thread,
                args=(request.get_response_sender(), input0, gpu_output),
            )

            thread.daemon = True

            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1

            thread.start()

    def finalize(self):
        inflight_threads = True
        cycles = 0
        logging_time_sec = 5
        sleep_time_sec = 0.1
        cycle_to_log = logging_time_sec / sleep_time_sec
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
                if cycles % cycle_to_log == 0:
                    print(
                        f"Waiting for {self.inflight_thread_count} response threads to complete..."
                    )
            if inflight_threads:
                time.sleep(sleep_time_sec)
                cycles += 1
