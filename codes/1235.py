import os

import numpy as np
import tensorrt as trt
import torch
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"

        self.logger = trt.Logger(trt.Logger.ERROR)
        engine_path = os.getenv("TRT_ENGINE_LOCATION")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True

            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if shape[0] < 0:
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)

                self.context.set_input_shape(name, profile_shape[0])
                shape = self.context.get_tensor_shape(name)

            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": None,
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def execute(self, requests):

        responses = []
        for request in requests:
            allocations = []
            output = torch.asarray(
                np.zeros(self.outputs[0]["shape"], self.outputs[0]["dtype"]),
                device=self.device,
            )
            input_tensor = torch.asarray(
                pb_utils.get_input_tensor_by_name(request, "image").as_numpy(),
                device=self.device,
            )
            self.inputs[0]["allocation"] = input_tensor.data_ptr()
            allocations.append(input_tensor.data_ptr())
            self.outputs[0]["allocation"] = output.data_ptr()
            allocations.append(output.data_ptr())
            self.context.execute_v2(allocations)
            out_tensor = pb_utils.Tensor.from_dlpack("features", output.cpu())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
            self.inputs[0]["allocation"] = None
            self.outputs[0]["allocation"] = None
        return responses
