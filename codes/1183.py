import triton_python_backend_utils as pb_utils
from model_common import ResponseSenderModelCommon


class TritonPythonModel:
    def initialize(self, args):
        self._common = ResponseSenderModelCommon(pb_utils)

    async def execute(self, requests):
        return self._common.execute(requests, use_async=True)
