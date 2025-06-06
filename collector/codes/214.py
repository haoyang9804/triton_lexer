














from abc import ABCMeta, abstractmethod

import tritonclient.grpc.model_config_pb2 as model_config

from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema
from merlin.systems.triton.export import _convert_dtype


class TritonOperator(BaseOperator, metaclass=ABCMeta):
    

    def __init__(self, base_op: BaseOperator):
        
        super().__init__()
        self.op = base_op

    @property
    def export_name(self):
        
        return self.__class__.__name__.lower()

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        
        return transformable

    @abstractmethod
    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ):
        


def add_model_param(params, paramclass, col_schema, dims=None):
    if col_schema.is_list and col_schema.is_ragged:
        params.append(
            paramclass(
                name=col_schema.name + "__values",
                data_type=_convert_dtype(col_schema.dtype),
                dims=dims[1:],
            )
        )
        params.append(
            paramclass(
                name=col_schema.name + "__offsets", data_type=model_config.TYPE_INT32, dims=[-1]
            )
        )
    else:
        params.append(
            paramclass(name=col_schema.name, data_type=_convert_dtype(col_schema.dtype), dims=dims)
        )
