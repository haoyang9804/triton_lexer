import importlib.resources
import json
import os
import pathlib
from shutil import copyfile

import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.runtimes.triton.ops.operator import TritonOperator
from merlin.systems.triton.conversions import (
    tensor_table_to_triton_request,
    triton_response_to_tensor_table,
)
from merlin.systems.triton.export import _add_model_param


class TransformWorkflowTriton(TritonOperator):

    def __init__(self, op):

        super().__init__(op)

        self._nvt_model_name = None

        if op.workflow is not None:
            self.input_schema = op.workflow.input_schema
            self.output_schema = op.workflow.output_schema

    def transform(self, col_selector: ColumnSelector, transformable: Transformable):

        inference_request = tensor_table_to_triton_request(
            self._nvt_model_name,
            transformable,
            self.input_schema,
            self.output_schema,
        )

        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise RuntimeError(inference_response.error().message())

        response_table = triton_response_to_tensor_table(
            inference_response, type(transformable), self.output_schema
        )

        return response_table

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "TransformWorkflowTriton":

        input_column_schemas = [
            ColumnSchema(name, **schema_properties)
            for name, schema_properties in json.loads(config["input_dict"]).items()
        ]
        input_schema = Schema(input_column_schemas)

        output_column_schemas = [
            ColumnSchema(name, **schema_properties)
            for name, schema_properties in json.loads(config["output_dict"]).items()
        ]
        output_schema = Schema(output_column_schemas)
        cls_instance = cls(None)
        cls_instance.input_schema = input_schema
        cls_instance.output_schema = output_schema

        params = json.loads(config["params"])
        cls_instance.set_nvt_model_name(params["nvt_model_name"])

        return cls_instance

    @property
    def nvt_model_name(self):

        return self._nvt_model_name

    def set_nvt_model_name(self, nvt_model_name):

        self._nvt_model_name = nvt_model_name

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:

        return self.op.workflow.output_schema

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ):

        modified_workflow = self.op.workflow.remove_inputs(self.op.label_columns)
        export_name = self.__class__.__name__.lower()
        node_name = f"{node_id}_{export_name}" if node_id is not None else export_name
        self.set_nvt_model_name(node_name)
        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(parents=True, exist_ok=True)

        backend_model_config = _generate_nvtabular_model(
            modified_workflow,
            node_name,
            node_export_path,
            sparse_max=self.op.sparse_max,
            max_batch_size=self.op.max_batch_size,
            cats=self.op.cats,
            conts=self.op.conts,
        )

        return backend_model_config


def _generate_nvtabular_model(
    workflow,
    name,
    output_path,
    version=1,
    max_batch_size=None,
    sparse_max=None,
    backend="python",
    cats=None,
    conts=None,
):

    workflow.save(os.path.join(output_path, str(version), "workflow"))
    config = _generate_nvtabular_config(
        workflow,
        name,
        output_path,
        max_batch_size,
        sparse_max=sparse_max,
        backend=backend,
        cats=cats,
        conts=conts,
    )

    with importlib.resources.path(
        "merlin.systems.triton.models", "workflow_model.py"
    ) as workflow_model:
        copyfile(
            workflow_model,
            os.path.join(output_path, str(version), "model.py"),
        )

    return config


def _generate_nvtabular_config(
    workflow,
    name,
    output_path,
    max_batch_size=None,
    sparse_max=None,
    backend="python",
    cats=None,
    conts=None,
):

    config = model_config.ModelConfig(
        name=name,
        backend=backend,
        max_batch_size=max_batch_size,
        instance_group=[
            model_config.ModelInstanceGroup(
                kind=model_config.ModelInstanceGroup.Kind.KIND_AUTO
            )
        ],
    )

    config.parameters["python_module"].string_value = (
        "merlin.systems.triton.models.workflow_model"
    )

    config.parameters["cats"].string_value = json.dumps(cats) if cats else ""
    config.parameters["conts"].string_value = json.dumps(conts) if conts else ""

    if sparse_max:

        config.parameters["sparse_max"].string_value = json.dumps(sparse_max)

    for col_name, col_schema in workflow.input_schema.column_schemas.items():
        _add_model_param(col_schema, model_config.ModelInput, config.input)

    for col_name, col_schema in workflow.output_schema.column_schemas.items():
        if sparse_max and col_name in sparse_max.keys():

            dim = sparse_max[col_name]
            _add_model_param(
                col_schema, model_config.ModelOutput, config.output, [-1, dim]
            )
        else:
            _add_model_param(col_schema, model_config.ModelOutput, config.output)

    with open(os.path.join(output_path, "config.pbtxt"), "w", encoding="utf-8") as o:
        text_format.PrintMessage(config, o)
    return config
