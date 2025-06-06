import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        logger = pb_utils.Logger
        logger.log("Initialize-Specific Msg!", logger.INFO)
        logger.log_info("Initialize-Info Msg!")
        logger.log_warn("Initialize-Warning Msg!")
        logger.log_error("Initialize-Error Msg!")
        logger.log_verbose("Initialize-Verbose Msg!")

    def execute(self, requests):

        logger = pb_utils.Logger
        logger.log("Execute-Specific Msg!", logger.INFO)
        logger.log_info("Execute-Info Msg!")
        logger.log_warn("Execute-Warning Msg!")
        logger.log_error("Execute-Error Msg!")
        logger.log_verbose("Execute-Verbose Msg!")

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))

        logger.log("Execute-Specific Msg!", logger.INFO)
        logger.log_info("Execute-Info Msg!")
        logger.log_warn("Execute-Warning Msg!")
        logger.log_error("Execute-Error Msg!")
        logger.log_verbose("Execute-Verbose Msg!")

        return responses

    def finalize(self):
        logger = pb_utils.Logger
        logger.log("Finalize-Specific Msg!", logger.INFO)
        logger.log_info("Finalize-Info Msg!")
        logger.log_warn("Finalize-Warning Msg!")
        logger.log_error("Finalize-Error Msg!")
        logger.log_verbose("Finalize-Verbose Msg!")
