import json
import os

import pandas as pd


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

import merlin.dtypes as md
from merlin.core.dispatch import is_string_dtype, make_df
from merlin.systems.dag.ops import compute_dims
from merlin.table import NumpyColumn, TensorTable


def convert_table_to_triton_input(
    schema, batch, input_class=grpcclient.InferInput, dtype="int32"
):

    cpu_table = batch.cpu()

    numpy_cols = {}
    for col_name, col_values in cpu_table.items():
        np_col = (
            NumpyColumn(col_values)
            if not isinstance(col_values, NumpyColumn)
            else col_values
        )
        numpy_cols[col_name] = np_col
    numpy_table = TensorTable(numpy_cols)

    inputs = []
    for col_name, col_values in numpy_table.to_dict().items():
        triton_input = _convert_array_to_triton_input(col_name, col_values, input_class)
        inputs.append(triton_input)

    return inputs


def convert_df_to_triton_input(
    schema, batch, input_class=grpcclient.InferInput, dtype="int32"
):

    df_dict = _convert_df_to_dict(schema, batch, dtype)
    inputs = [
        _convert_array_to_triton_input(col_name, col_values, input_class)
        for col_name, col_values in df_dict.items()
    ]
    return inputs


def _convert_array_to_triton_input(
    col_name, col_values, input_class=grpcclient.InferInput
):

    col_dtype = md.dtype(col_values.dtype).to_numpy
    dtype = np_to_triton_dtype(col_dtype)
    input_tensor = input_class(col_name, col_values.shape, dtype)

    col_values = col_values.astype(col_dtype)
    input_tensor.set_data_from_numpy(col_values)

    return input_tensor


def convert_triton_output_to_df(columns, response):

    return make_df({col: response.as_numpy(col) for col in columns})


def get_column_types(path):

    path = os.path.join(path, "column_types.json")
    return json.load(open(path, encoding="utf-8"))


def _convert_tensor(t):
    out = t.as_numpy()
    if len(out.shape) == 2:
        out = out[:, 0]

    if is_string_dtype(out.dtype):
        out = out.astype("str")
    return out


def _convert_df_to_dict(schema, batch, dtype="int32"):
    df_dict = {}
    for col_name, col_schema in schema.column_schemas.items():
        col = batch[col_name]
        shape = compute_dims(col_schema)
        shape[0] = len(col)

        if col_schema.is_list:
            if isinstance(col, pd.Series):
                raise ValueError("this function doesn't support CPU list values yet")

            if col_schema.is_ragged:
                df_dict[col_name + "__values"] = col.list.leaves.values_host.astype(
                    col_schema.dtype.to_numpy
                )
                offsets = col._column.offsets.values_host.astype(dtype)
                df_dict[col_name + "__offsets"] = offsets
            else:
                values = col.list.leaves.values_host
                values = values.reshape(*shape).astype(col_schema.dtype.to_numpy)
                df_dict[col_name] = values

        else:
            values = col.values if isinstance(col, pd.Series) else col.values_host
            values = values.reshape(*shape).astype(col_schema.dtype.to_numpy)
            df_dict[col_name] = values
    return df_dict
