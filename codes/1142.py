import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {"name": "MODEL_NAME", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "INPUT0", "data_type": "TYPE_INT32", "dims": [1, 16]},
            {"name": "INPUT1", "data_type": "TYPE_INT32", "dims": [1, 16]},
        ]
        outputs = [
            {"name": "OUTPUT0", "data_type": "TYPE_INT32", "dims": [16]},
            {"name": "OUTPUT1", "data_type": "TYPE_INT32", "dims": [16]},
        ]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        for input in inputs:
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            model_name = pb_utils.get_input_tensor_by_name(request, "MODEL_NAME")
            model_name_string = model_name.as_numpy()[0]

            infer_request = pb_utils.InferenceRequest(
                model_name=model_name_string,
                requested_output_names=["OUTPUT0", "OUTPUT1"],
                inputs=[in_0, in_1],
                trace=request.trace(),
            )

            infer_response = infer_request.exec()

            inference_response = pb_utils.InferenceResponse(
                output_tensors=infer_response.output_tensors()
            )
            responses.append(inference_response)

        return responses
