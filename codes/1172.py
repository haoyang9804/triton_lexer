import file1
import triton_python_backend_utils as pb_utils

from . import file2


class TritonPythonModel:
    def initialize(self, args):
        if file1.FILE_NAME != "FILE1" or file2.FILE_NAME != "FILE2":
            raise pb_utils.TritonModelException("Imports do not work")

    def execute(self, requests):
        pass
