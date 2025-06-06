from os import path

import c_python_backend_utils as c_utils
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):

        file_name = "free_memory.txt"
        current_free_memory = str(c_utils.shared_memory.free_memory())
        if path.exists(file_name):
            with open(file_name, "r") as f:
                expected_free_memory = f.read()
                assert expected_free_memory == current_free_memory, (
                    f"Free shared memory before and after restart are not equal. "
                    "{expected_free_memory} (before) != {current_free_memory} (after)."
                )
        else:
            with open(file_name, "w") as f:
                f.write(current_free_memory)

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
