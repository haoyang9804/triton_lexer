import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


class TritonPythonModel:

    def initialize(self, args):
        self._model_name = args["model_name"]

    def execute(self, requests):
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            gpu_output = pb_utils.get_input_tensor_by_name(
                request, "GPU_OUTPUT"
            ).as_numpy()

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
                    outptu0_pytorch = from_dlpack(input0.to_dlpack()).cpu()
                    output0 = pb_utils.Tensor.from_dlpack(
                        "OUTPUT0", to_dlpack(outptu0_pytorch)
                    )

            next_gpu_output = pb_utils.Tensor("NEXT_GPU_OUTPUT", gpu_output[1:])

            if self._model_name != "dlpack_io_identity_1":
                infer_request = pb_utils.InferenceRequest(
                    model_name="dlpack_io_identity_1",
                    inputs=[
                        input0,
                        pb_utils.get_input_tensor_by_name(request, "GPU_OUTPUT"),
                    ],
                    requested_output_names=["OUTPUT0"],
                )
                infer_response = infer_request.exec()

                if infer_response.has_error():
                    raise pb_utils.TritonModelException(
                        infer_response.error().message()
                    )

                bls_output0 = pb_utils.get_output_tensor_by_name(
                    infer_response, "OUTPUT0"
                )
                if not output0.is_cpu():
                    bls_output0 = (
                        from_dlpack(bls_output0.to_dlpack()).detach().cpu().numpy()
                    )
                else:
                    bls_output0 = bls_output0.as_numpy()

                if not input0.is_cpu():
                    input0 = from_dlpack(input0.to_dlpack()).detach().cpu().numpy()
                else:
                    input0 = input0.as_numpy()

                if not np.allclose(bls_output0, input0):
                    raise pb_utils.TritonModelException(
                        "BLS input and output tensors are not equal"
                    )

            responses.append(pb_utils.InferenceResponse([output0, next_gpu_output]))

        return responses
