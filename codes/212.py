import importlib.resources
import os
import pathlib
from shutil import copyfile
from typing import List, Tuple

import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

from merlin.dag import postorder_iter_nodes
from merlin.dag.runtime import Runtime
from merlin.systems.dag.ops import compute_dims
from merlin.systems.dag.ops.compat import (
    cuml_ensemble,
    lightgbm,
    sklearn_ensemble,
    treelite_sklearn,
    xgboost,
)
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.runtimes.triton.ops.operator import (
    TritonOperator,
    add_model_param,
)
from merlin.systems.dag.runtimes.triton.ops.workflow import TransformWorkflowTriton

tensorflow = None
try:
    from nvtabular.loader.tf_utils import configure_tensorflow

    configure_tensorflow()
    import tensorflow
except ImportError:
    ...

torch = None
try:
    import torch
except ImportError:
    ...


TRITON_OP_TABLE = {}
TRITON_OP_TABLE[TransformWorkflow] = TransformWorkflowTriton

if cuml_ensemble or lightgbm or sklearn_ensemble or treelite_sklearn or xgboost:
    from merlin.systems.dag.ops.fil import PredictForest
    from merlin.systems.dag.runtimes.triton.ops.fil import PredictForestTriton

    TRITON_OP_TABLE[PredictForest] = PredictForestTriton

if tensorflow:
    from merlin.systems.dag.ops.tensorflow import PredictTensorflow
    from merlin.systems.dag.runtimes.triton.ops.tensorflow import (
        PredictTensorflowTriton,
    )

    TRITON_OP_TABLE[PredictTensorflow] = PredictTensorflowTriton

if torch:
    from merlin.systems.dag.ops.pytorch import PredictPyTorch
    from merlin.systems.dag.runtimes.triton.ops.pytorch import PredictPyTorchTriton

    TRITON_OP_TABLE[PredictPyTorch] = PredictPyTorchTriton


class TritonExecutorRuntime(Runtime):

    def __init__(self):
        super().__init__()
        self.op_table = TRITON_OP_TABLE

    def export(
        self, ensemble, path: str, version: int = 1, name: str = None
    ) -> Tuple[model_config.ModelConfig, List[model_config.ModelConfig]]:

        triton_model_name = name or "executor_model"

        nodes = list(
            postorder_iter_nodes(ensemble.graph.output_node, flatten_subgraphs=True)
        )

        for node in nodes:
            if type(node.op) in self.op_table:
                node.op = self.op_table[type(node.op)](node.op)

        node_id_table, _ = _create_node_table(nodes)

        artifact_path = (
            pathlib.Path(path) / triton_model_name / str(version) / "ensemble"
        )
        artifact_path.mkdir(parents=True, exist_ok=True)

        node_configs = []
        for node in nodes:
            node_id = node_id_table.get(node, None)
            if node_id is not None:
                node_config = node.op.export(
                    path,
                    node.input_schema,
                    node.output_schema,
                    node_id=node_id,
                    version=version,
                )
                if node_config is not None:
                    node_configs.append(node_config)

            node.op.save_artifacts(str(artifact_path))

        executor_config = self._executor_model_export(path, triton_model_name, ensemble)

        return (executor_config, node_configs)

    def _executor_model_export(
        self,
        path: str,
        export_name: str,
        ensemble,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ) -> model_config.ModelConfig:

        params = params or {}

        node_name = f"{node_id}_{export_name}" if node_id is not None else export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(parents=True, exist_ok=True)

        config = model_config.ModelConfig(
            name=node_name,
            backend="python",
            platform="merlin_executor",
            instance_group=[
                model_config.ModelInstanceGroup(
                    kind=model_config.ModelInstanceGroup.Kind.KIND_AUTO
                )
            ],
        )

        input_schema = ensemble.input_schema
        output_schema = ensemble.output_schema

        for col_schema in input_schema.column_schemas.values():
            col_dims = compute_dims(col_schema)
            add_model_param(config.input, model_config.ModelInput, col_schema, col_dims)

        for col_schema in output_schema.column_schemas.values():
            col_dims = compute_dims(col_schema)
            add_model_param(
                config.output, model_config.ModelOutput, col_schema, col_dims
            )

        with open(
            os.path.join(node_export_path, "config.pbtxt"), "w", encoding="utf-8"
        ) as o:
            text_format.PrintMessage(config, o)

        os.makedirs(node_export_path, exist_ok=True)
        os.makedirs(os.path.join(node_export_path, str(version)), exist_ok=True)
        with importlib.resources.path(
            "merlin.systems.triton.models", "executor_model.py"
        ) as executor_model:
            copyfile(
                executor_model,
                os.path.join(node_export_path, str(version), "model.py"),
            )

        ensemble.save(os.path.join(node_export_path, str(version), "ensemble"))

        return config


def _create_node_table(nodes):
    exportable_node_idx = 0
    node_id_lookup = {}
    for node in nodes:
        if isinstance(node.op, TritonOperator):
            node_id_lookup[node] = exportable_node_idx
            exportable_node_idx += 1

    return node_id_lookup, exportable_node_idx
