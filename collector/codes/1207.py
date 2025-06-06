import json
import traceback

import triton_python_backend_utils as pb_utils
from lib.triton_decoder import TritonDecoder


def get_valid_param_value(param, default_value=""):
    value = param.get("string_value", "")
    return default_value if value.startswith("${") or value == "" else value


class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])

        params = model_config["parameters"]

        accumulate_tokens_str = get_valid_param_value(
            params.get("accumulate_tokens", {})
        )
        self.accumulate_tokens = accumulate_tokens_str.lower() in [
            "true",
            "yes",
            "1",
            "t",
        ]

        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(model_config)

        self.logger = pb_utils.Logger

        default_tensorrt_llm_model_name = "tensorrt_llm"
        self.llm_model_name = get_valid_param_value(
            params.get("tensorrt_llm_model_name", {}), default_tensorrt_llm_model_name
        )

        self.draft_llm_model_name = get_valid_param_value(
            params.get("tensorrt_llm_draft_model_name", {}), None
        )

        self.multimodal_encoders_name = get_valid_param_value(
            params.get("multimodal_encoders_name", {}), None
        )

        self.decoder = TritonDecoder(
            streaming=self.decoupled,
            accumulate=self.accumulate_tokens,
            preproc_model_name="preprocessing",
            postproc_model_name="postprocessing",
            llm_model_name=self.llm_model_name,
            draft_llm_model_name=self.draft_llm_model_name,
            multimodal_encoders_name=self.multimodal_encoders_name,
        )

    def execute(self, requests):

        responses = []

        for request in requests:
            if self.decoupled:
                response_sender = request.get_response_sender()
            try:

                req = self.decoder.convert_triton_request(request)
                req.validate()
                speculative_decode = (
                    req.num_draft_tokens is not None and req.num_draft_tokens[0][0] > 0
                )
                if speculative_decode and (
                    self.draft_llm_model_name is None or self.draft_llm_model_name == ""
                ):
                    raise Exception(
                        "cannot perform speculative decoding without draft model"
                    )
                is_multimodal = (
                    req.image_input is not None
                    or req.image_bytes_input is not None
                    or req.image_url_input is not None
                    or req.video_bytes_input is not None
                )

                if speculative_decode and is_multimodal:
                    raise Exception(
                        "Multimodal and speculative decoding is not currently supported"
                    )
                res_gen = self.decoder.decode(
                    req,
                    speculative_decoding=speculative_decode,
                    is_multimodal=is_multimodal,
                )

                for res in res_gen:
                    triton_response = self.decoder.create_triton_response(res)
                    if self.decoupled:
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

                if self.decoupled:
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )

            except Exception:
                self.logger.log_error(traceback.format_exc())

                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(traceback.format_exc()),
                )

                if self.decoupled:
                    response_sender.send(error_response)
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    responses.append(error_response)

            self.decoder.reset_decoder()

        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses

    def finalize(self):

        print("Cleaning up...")
