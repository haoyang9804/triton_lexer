import json
import pathlib

import triton_python_backend_utils as pb_utils

import nvtabular
from merlin.systems.triton import _convert_tensor
from merlin.systems.triton.utils import triton_error_handling, triton_multi_request
from merlin.systems.workflow.base import WorkflowRunner


class TritonPythonModel:

    def initialize(self, args):

        model_repo = args["model_repository"]
        repository_path = pathlib.Path(model_repo)

        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent

        workflow_path = repository_path / str(args["model_version"]) / "workflow"
        model_device = args["model_instance_kind"]

        self.workflow = nvtabular.Workflow.load(str(workflow_path))

        self.model_config = json.loads(args["model_config"])

        input_dtypes = self.workflow.input_dtypes.items()
        self.input_dtypes, self.input_multihots = self._parse_input_dtypes(input_dtypes)

        self.output_dtypes = {}
        for col_name, col_schema in self.workflow.output_schema.column_schemas.items():
            if col_schema.is_list and col_schema.is_ragged:
                self._set_output_dtype(col_name + "__offsets")
                self._set_output_dtype(col_name + "__values")
            else:
                self._set_output_dtype(col_name)

        self.runner = WorkflowRunner(
            self.workflow, self.output_dtypes, self.model_config, model_device
        )

    def _set_output_dtype(self, name):
        conf = pb_utils.get_output_config_by_name(self.model_config, name)
        self.output_dtypes[name] = pb_utils.triton_string_to_numpy(conf["data_type"])

    @triton_multi_request
    @triton_error_handling
    def execute(self, request):

        try:

            input_tensors = {
                name: _convert_tensor(pb_utils.get_input_tensor_by_name(request, name))
                for name in self.input_dtypes
            }

            for name, dtype in self.input_multihots.items():
                values = _convert_tensor(
                    pb_utils.get_input_tensor_by_name(request, name + "__values")
                )
                offsets = _convert_tensor(
                    pb_utils.get_input_tensor_by_name(request, name + "__offsets")
                )
                input_tensors[name] = (values, offsets)

            transformed = self.runner.run_workflow(input_tensors)
            result = [pb_utils.Tensor(name, data) for name, data in transformed.items()]

            return pb_utils.InferenceResponse(result)

        except Exception as exc:
            import traceback

            raise pb_utils.TritonModelException(
                f"Error: {type(exc)} - {str(exc)}, "
                f"Traceback: {traceback.format_tb(exc.__traceback__)}"
            ) from exc

    def _is_list_dtype(self, column: str) -> bool:

        col_schema = self.workflow.input_schema.get(column)
        if col_schema is None:
            return False
        return col_schema.is_list and col_schema.is_ragged

    def _parse_input_dtypes(self, dtypes):
        input_dtypes = {
            col: dtype for col, dtype in dtypes if not self._is_list_dtype(col)
        }
        input_multihots = {
            col: dtype for col, dtype in dtypes if self._is_list_dtype(col)
        }

        return input_dtypes, input_multihots
