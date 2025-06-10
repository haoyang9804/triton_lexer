import triton_python_backend_utils as pb_utils
from cuda import cuda


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        input = {"name": "INPUT", "data_type": "TYPE_FP32", "dims": [1]}
        output = {"name": "OUTPUT", "data_type": "TYPE_FP32", "dims": [1]}

        auto_complete_model_config.set_max_batch_size(0)
        auto_complete_model_config.add_input(input)
        auto_complete_model_config.add_output(output)

        return auto_complete_model_config

    def initialize(self, args):
        self.mem_ptr = None

        cuda.cuInit(0)
        cuda.cuCtxCreate(0, 0)

        mem_info = cuda.cuMemGetInfo()
        if mem_info[0] != 0:
            raise pb_utils.TritonModelException("Failed to get CUDA memory info")

        mem_alloc = cuda.cuMemAlloc(mem_info[2] * 0.4)
        if mem_alloc[0] != 0:
            raise pb_utils.TritonModelException("Failed to allocate CUDA memory")
        self.mem_ptr = mem_alloc[1]

    def finalize(self):
        if self.mem_ptr is not None:
            cuda.cuMemFree(self.mem_ptr)

    def execute(self, requests):

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
