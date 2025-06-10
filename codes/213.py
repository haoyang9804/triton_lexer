import pathlib

import numpy as np
import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import Schema
from merlin.systems.dag.runtimes.triton.ops.operator import TritonOperator
from merlin.systems.triton.conversions import (
    tensor_table_to_triton_request,
    triton_response_to_tensor_table,
)
from merlin.table import TensorTable


class PredictForestTriton(TritonOperator):

    def __init__(self, op, input_schema=None):

        super().__init__(op)
        if op is not None:
            self.fil_op = FILTriton(op.fil_op)
            self.backend = op.backend
            self.input_schema = op.input_schema
        if input_schema is not None:
            self.input_schema = input_schema
        self._fil_model_name = None

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:

        return self.fil_op.compute_output_schema(
            input_schema, col_selector, prev_output_schema=prev_output_schema
        )

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:

        return self.input_schema

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ):

        fil_model_config = self.fil_op.export(
            path,
            input_schema,
            output_schema,
            params=params,
            node_id=node_id,
            version=version,
        )
        self.set_fil_model_name(fil_model_config.name)
        params = params or {}
        params = {**params, "fil_model_name": self.fil_model_name}
        return super().export(
            path,
            input_schema,
            output_schema,
            params=params,
            node_id=node_id,
            version=version,
        )

    @property
    def fil_model_name(self):
        return self._fil_model_name

    def set_fil_model_name(self, fil_model_name):
        self._fil_model_name = fil_model_name

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:

        input0 = (
            np.array([column.values.ravel() for column in transformable.values()])
            .astype(np.float32)
            .T
        )

        inputs = TensorTable({"input__0": input0})
        input_schema = Schema(["input__0"])
        output_schema = Schema(["output__0"])

        inference_request = tensor_table_to_triton_request(
            self.fil_model_name, inputs, input_schema, output_schema
        )
        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise RuntimeError(str(inference_response.error().message()))

        return triton_response_to_tensor_table(
            inference_response, type(inputs), output_schema
        )


class FILTriton(TritonOperator):

    def __init__(self, op):

        self.max_batch_size = op.max_batch_size
        self.parameters = dict(**op.parameters)
        self.fil_model_class = op.fil_model_class
        super().__init__(op)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "fil_model"}

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:

        return self.op.compute_input_schema(
            root_schema, parents_schema, deps_schema, selector
        )

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:

        return self.op.compute_output_schema(
            input_schema, col_selector, prev_output_schema
        )

    def export(
        self,
        path,
        input_schema,
        output_schema,
        params: dict = None,
        node_id=None,
        version=1,
    ):

        node_name = (
            f"{node_id}_{self.export_name}" if node_id is not None else self.export_name
        )
        node_export_path = pathlib.Path(path) / node_name
        version_path = node_export_path / str(version)
        version_path.mkdir(parents=True, exist_ok=True)

        self.fil_model_class.save(version_path)

        config = fil_config(
            node_name,
            self.fil_model_class.model_type,
            self.fil_model_class.num_features,
            self.fil_model_class.num_classes,
            max_batch_size=self.max_batch_size,
            **self.parameters,
        )

        with open(node_export_path / "config.pbtxt", "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)

        return config


def fil_config(
    name,
    model_type,
    num_features,
    num_classes,
    *,
    max_batch_size=8192,
    predict_proba=False,
    output_class=False,
    threshold=0.5,
    algo="ALGO_AUTO",
    storage_type="AUTO",
    blocks_per_sm=0,
    threads_per_tree=1,
    transfer_threshold=0,
    instance_group="AUTO",
) -> model_config.ModelConfig:

    input_dim = num_features
    output_dim = 1

    if num_classes > 2:
        output_class = True

    if output_class and predict_proba:
        output_dim = num_classes

        output_dim = max(output_dim, 2)

    parameters = {
        "model_type": model_type,
        "predict_proba": "true" if predict_proba else "false",
        "output_class": "true" if output_class else "false",
        "threshold": f"{threshold:.4f}",
        "storage_type": storage_type,
        "algo": algo,
        "use_experimental_optimizations": "false",
        "blocks_per_sm": f"{blocks_per_sm:d}",
        "threads_per_tree": f"{threads_per_tree:d}",
        "transfer_threshold": f"{transfer_threshold:d}",
    }

    supported_instance_groups = {"auto", "cpu", "gpu"}
    instance_group = (
        instance_group.lower() if isinstance(instance_group, str) else instance_group
    )
    if instance_group == "auto":
        instance_group_kind = model_config.ModelInstanceGroup.Kind.KIND_AUTO
    elif instance_group == "cpu":
        instance_group_kind = model_config.ModelInstanceGroup.Kind.KIND_CPU
    elif instance_group == "gpu":
        instance_group_kind = model_config.ModelInstanceGroup.Kind.KIND_GPU
    else:
        raise ValueError(f"instance_group must be one of {supported_instance_groups}")

    config = model_config.ModelConfig(
        name=name,
        backend="fil",
        max_batch_size=max_batch_size,
        input=[
            model_config.ModelInput(
                name="input__0",
                data_type=model_config.TYPE_FP32,
                dims=[input_dim],
            )
        ],
        output=[
            model_config.ModelOutput(
                name="output__0", data_type=model_config.TYPE_FP32, dims=[output_dim]
            )
        ],
        instance_group=[model_config.ModelInstanceGroup(kind=instance_group_kind)],
    )

    for parameter_key, parameter_value in parameters.items():
        config.parameters[parameter_key].string_value = parameter_value

    return config
