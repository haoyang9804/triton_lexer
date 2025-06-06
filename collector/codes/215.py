import os
import pathlib
from shutil import copyfile

import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import Schema
from merlin.systems.dag.ops import compute_dims
from merlin.systems.dag.runtimes.triton.ops.operator import add_model_param
from merlin.systems.dag.runtimes.triton.ops.operator import TritonOperator
from merlin.systems.triton.conversions import (
    tensor_table_to_triton_request,
    triton_response_to_tensor_table,
)


class PredictPyTorchTriton(TritonOperator):

    def __init__(self, op):

        super().__init__(op)
        self._torch_model_name = None
        self.input_schema = op.input_schema
        self.output_schema = op.output_schema

    @property
    def torch_model_name(self):
        return self._torch_model_name

    def set_torch_model_name(self, torch_model_name):

        self._torch_model_name = torch_model_name

    def transform(self, col_selector: ColumnSelector, transformable: Transformable):
        inference_request = tensor_table_to_triton_request(
            self.torch_model_name,
            transformable,
            self.input_schema,
            self.output_schema,
        )

        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise RuntimeError(str(inference_response.error().message()))

        return triton_response_to_tensor_table(
            inference_response, type(transformable), self.output_schema
        )

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:

        return self.input_schema

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:

        return self.output_schema

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ):

        export_name = self.__class__.__name__.lower()
        node_name = f"{node_id}_{export_name}" if node_id is not None else export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        export_model_path = pathlib.Path(node_export_path) / str(version)
        export_model_path.mkdir(exist_ok=True)

        if self.op.path:
            copyfile(
                str(self.op.path),
                export_model_path / "model.pt",
            )
        else:
            self.op.model.save(export_model_path / "model.pt")

        self.set_torch_model_name(node_name)
        backend_model_config = self._export_model_config(node_name, node_export_path)
        return backend_model_config

    def _export_model_config(self, name, output_path):

        config = self._export_torchscript_config(name, output_path)

        return config

    def _export_torchscript_config(self, name, output_path):

        config = model_config.ModelConfig(
            name=name,
            instance_group=[
                model_config.ModelInstanceGroup(
                    kind=model_config.ModelInstanceGroup.Kind.KIND_AUTO
                )
            ],
        )

        config.backend = "pytorch"
        config.platform = "pytorch_libtorch"
        config.parameters["INFERENCE_MODE"].string_value = "true"

        for _, col_schema in self.input_schema.column_schemas.items():
            add_model_param(
                config.input,
                model_config.ModelInput,
                col_schema,
                compute_dims(col_schema),
            )

        for _, col_schema in self.output_schema.column_schemas.items():
            add_model_param(
                config.output,
                model_config.ModelOutput,
                col_schema,
                compute_dims(col_schema),
            )

        with open(
            os.path.join(output_path, "config.pbtxt"), "w", encoding="utf-8"
        ) as o:
            text_format.PrintMessage(config, o)
        return config

    @property
    def scalar_shape(self):
        return []
