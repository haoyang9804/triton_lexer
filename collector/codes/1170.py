import os

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):

        if "MY_ENV" not in os.environ or os.environ["MY_ENV"] != "MY_ENV":
            raise pb_utils.TritonModelException(
                "MY_ENV doesn't exists or contains incorrect value"
            )

    def execute(self, requests):
        pass
