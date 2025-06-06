import os

import numpy as np
import triton_python_backend_utils as pb_utils


async def _send_identity_tensor(size, is_decoupled):
    tensor_size = [1, size]
    input0_np = np.random.randn(*tensor_size)
    input0 = pb_utils.Tensor("INPUT0", input0_np.astype(np.float32))
    infer_request = pb_utils.InferenceRequest(
        model_name="identity_fp32", inputs=[input0], requested_output_names=["OUTPUT0"]
    )

    if is_decoupled:
        infer_responses = await infer_request.async_exec(decoupled=True)
        infer_response = next(infer_responses)
    else:
        infer_response = await infer_request.async_exec()

    return input0_np, infer_response


async def test_bls_out_of_memory():
    is_decoupled = True if os.environ["BLS_KIND"] == "decoupled" else False

    tensor_size = 256 * 1024 * 1024
    input0_np, infer_response = await _send_identity_tensor(tensor_size, is_decoupled)

    out_of_memory_message = "Failed to increase the shared memory pool size for key"

    if infer_response.has_error():
        if not (out_of_memory_message in infer_response.error().message()):
            return False
    else:
        output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
        if output0 is None:
            return False
        if not np.allclose(output0.as_numpy(), input0_np):
            return False

    tensor_size = 50 * 1024 * 1024
    for _ in range(4):
        input0_np, infer_response = await _send_identity_tensor(
            tensor_size, is_decoupled
        )

        if infer_response.has_error():
            if not (out_of_memory_message in infer_response.error().message()):
                return False
        else:
            output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
            if output0 is None:
                return False
            if not np.allclose(output0.as_numpy(), input0_np):
                return False

    return True


class TritonPythonModel:
    async def execute(self, requests):
        responses = []
        for _ in requests:

            result = await test_bls_out_of_memory()
            responses.append(
                pb_utils.InferenceResponse(
                    [pb_utils.Tensor("OUTPUT0", np.array([result], dtype=np.float16))]
                )
            )
        return responses
