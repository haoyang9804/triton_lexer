import pathlib
from pathlib import Path

import triton_python_backend_utils as pb_utils

from merlin.dag import postorder_iter_nodes
from merlin.systems.dag import Ensemble
from merlin.systems.dag.runtimes.triton import TritonExecutorRuntime
from merlin.systems.triton.conversions import (
    tensor_table_to_triton_response,
    triton_request_to_tensor_table,
)
from merlin.systems.triton.utils import triton_error_handling, triton_multi_request


class TritonPythonModel:

    def initialize(self, args):

        model_repo = args["model_repository"]
        repository_path = _parse_model_repository(model_repo)

        ensemble_path = (
            Path(repository_path)
            / args["model_name"]
            / str(args["model_version"])
            / "ensemble"
        )

        self.ensemble = Ensemble.load(str(ensemble_path))

        for node in list(
            postorder_iter_nodes(
                self.ensemble.graph.output_node, flatten_subgraphs=True
            )
        ):
            node.op.load_artifacts(str(ensemble_path))

    @triton_multi_request
    @triton_error_handling
    def execute(self, request):

        inputs = triton_request_to_tensor_table(request, self.ensemble.input_schema)

        try:
            outputs = self.ensemble.transform(inputs, runtime=TritonExecutorRuntime())
        except Exception as exc:
            import traceback

            raise pb_utils.TritonModelException(traceback.format_exc()) from exc

        return tensor_table_to_triton_response(outputs, self.ensemble.output_schema)


def _parse_model_repository(model_repository: str) -> str:

    if model_repository.endswith(".py"):
        return str(pathlib.Path(model_repository).parent.parent.parent)
    else:
        return str(pathlib.Path(model_repository).parent)
