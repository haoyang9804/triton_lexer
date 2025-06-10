import os
import shutil

import numpy as np
import pytest

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format

from merlin.core.dispatch import make_df
from merlin.dag import ColumnSelector
from merlin.dag.ops.subgraph import Subgraph
from merlin.schema import Schema, Tags
from merlin.table import TensorTable
from nvtabular import Workflow
from nvtabular import ops as wf_ops

loader_tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")


loader_tf_utils.configure_tensorflow()
tf = pytest.importorskip("tensorflow")

triton = pytest.importorskip("merlin.systems.triton")
export = pytest.importorskip("merlin.systems.dag.ensemble")

import tritonclient.grpc.model_config_pb2 as model_config

from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.triton.utils import run_ensemble_on_tritonserver
from tests.unit.systems.utils.tf import create_tf_model

TRITON_SERVER_PATH = shutil.which("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_config_verification(tmpdir, dataset, engine):

    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[
            name
        ].with_tags([Tags.USER])
    selector = ColumnSelector(["x", "y", "id"])

    workflow_ops = selector >> wf_ops.Rename(postfix="_nvt")
    workflow = Workflow(workflow_ops["x_nvt"])
    workflow.fit(dataset)

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="x_nvt", dtype=tf.float64, shape=()),
            tf.keras.layers.Reshape((1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, name="output"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    triton_chain = (
        selector
        >> TransformWorkflow(workflow, cats=["x_nvt"])
        >> PredictTensorflow(model)
    )
    triton_ens = Ensemble(triton_chain, schema)

    ensemble_config, node_configs = triton_ens.export(str(tmpdir))

    config_path = tmpdir / ensemble_config.name / "config.pbtxt"

    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        assert parsed.name == "executor_model"
        assert parsed.platform == "merlin_executor"
        assert hasattr(parsed, "ensemble_scheduling")

    df = make_df({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "id": [7, 8, 9]})

    request_schema = Schema([schema["x"], schema["y"], schema["id"]])

    output_columns = triton_ens.output_schema.column_names
    response = run_ensemble_on_tritonserver(
        str(tmpdir), request_schema, df, output_columns, ensemble_config.name
    )
    assert len(response["output"]) == df.shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_multi_op_run(tmpdir, dataset, engine):

    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[
            name
        ].with_tags([Tags.USER])

    workflow_ops = ["name-cat"] >> wf_ops.Categorify(cat_cache="host")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    embedding_shapes_1 = wf_ops.get_embedding_sizes(workflow)

    cats = ["name-string"] >> wf_ops.Categorify(cat_cache="host")
    workflow_2 = Workflow(cats)
    workflow_2.fit(dataset)

    embedding_shapes = wf_ops.get_embedding_sizes(workflow_2)
    embedding_shapes_1.update(embedding_shapes)

    model = create_tf_model(["name-cat", "name-string"], [], embedding_shapes_1)

    triton_chain_1 = ["name-cat"] >> TransformWorkflow(workflow)
    triton_chain_2 = ["name-string"] >> TransformWorkflow(workflow_2)
    triton_chain = (triton_chain_1 + triton_chain_2) >> PredictTensorflow(model)

    triton_ens = Ensemble(triton_chain, schema)

    ensemble_config, nodes_config = triton_ens.export(str(tmpdir))
    config_path = tmpdir / "executor_model" / "config.pbtxt"

    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        assert parsed.name == "executor_model"
        assert parsed.platform == "merlin_executor"
        assert hasattr(parsed, "ensemble_scheduling")

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]
    request_schema = workflow.input_schema + workflow_2.input_schema

    response = run_ensemble_on_tritonserver(
        str(tmpdir), request_schema, df, ["predictions"], ensemble_config.name
    )
    assert len(response["predictions"]) == df.shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("python", [False, True])
def test_workflow_tf_python_wrapper(tmpdir, dataset, engine, python):

    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[
            name
        ].with_tags([Tags.USER])

    workflow_ops = ["name-cat"] >> wf_ops.Categorify(cat_cache="host")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    embedding_shapes_1 = wf_ops.get_embedding_sizes(workflow)

    cats = ["name-string"] >> wf_ops.Categorify(cat_cache="host")
    workflow_2 = Workflow(cats)
    workflow_2.fit(dataset)

    embedding_shapes = wf_ops.get_embedding_sizes(workflow_2)
    embedding_shapes_1.update(embedding_shapes)

    model = create_tf_model(["name-cat", "name-string"], [], embedding_shapes_1)

    triton_chain_1 = ["name-cat"] >> TransformWorkflow(workflow)
    triton_chain_2 = ["name-string"] >> TransformWorkflow(workflow_2)
    triton_chain = (triton_chain_1 + triton_chain_2) >> PredictTensorflow(model)

    triton_ens = Ensemble(triton_chain, schema)

    ensemble_config, nodes_config = triton_ens.export(str(tmpdir))
    config_path = tmpdir / "executor_model" / "config.pbtxt"

    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        assert parsed.name == "executor_model"
        assert parsed.platform == "merlin_executor"
        assert hasattr(parsed, "ensemble_scheduling")

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]
    request_schema = workflow.input_schema + workflow_2.input_schema

    response = run_ensemble_on_tritonserver(
        str(tmpdir), request_schema, df, ["predictions"], ensemble_config.name
    )
    assert len(response["predictions"]) == df.shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("python", [False, True])
def test_workflow_tf_subgraph_local(tmpdir, dataset, engine, python):

    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[
            name
        ].with_tags([Tags.USER])

    workflow_ops = ["name-cat"] >> wf_ops.Categorify(cat_cache="host")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)
    cat_df = workflow.transform(dataset).to_ddf().compute()
    embedding_shapes_1 = wf_ops.get_embedding_sizes(workflow)

    cats = ["name-string"] >> wf_ops.Categorify(cat_cache="host")
    workflow_2 = Workflow(cats)
    workflow_2.fit(dataset)
    string_df = workflow_2.transform(dataset).to_ddf().compute()

    embedding_shapes = wf_ops.get_embedding_sizes(workflow_2)
    embedding_shapes_1.update(embedding_shapes)

    model = create_tf_model(["name-cat", "name-string"], [], embedding_shapes_1)
    string_df["name-cat"] = cat_df["name-cat"]

    req_dict = (
        TensorTable.from_df(string_df[["name-string", "name-cat"]].iloc[:3])
        .cpu()
        .to_dict()
    )
    predictions = model.predict(req_dict)

    triton_chain_1 = Subgraph("cat", ["name-cat"] >> TransformWorkflow(workflow))
    triton_chain_2 = Subgraph(
        "string", ["name-string"] >> TransformWorkflow(workflow_2)
    )
    triton_chain = (triton_chain_1 + triton_chain_2) >> PredictTensorflow(model)

    triton_ens = Ensemble(triton_chain, schema)

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]

    response = triton_ens.transform(df)

    if hasattr(response, "to_pandas"):
        response = response.to_pandas()

    assert (
        np.stack(response["predictions"].to_numpy().tolist())
        == predictions["predictions"].tolist()
    ).all()
    assert len(response["predictions"]) == df.shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("python", [False, True])
def test_workflow_tf_subgraph_triton(tmpdir, dataset, engine, python):

    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[
            name
        ].with_tags([Tags.USER])

    workflow_ops = ["name-cat"] >> wf_ops.Categorify(cat_cache="host")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)
    cat_df = workflow.transform(dataset).to_ddf().compute()
    embedding_shapes_1 = wf_ops.get_embedding_sizes(workflow)

    cats = ["name-string"] >> wf_ops.Categorify(cat_cache="host")
    workflow_2 = Workflow(cats)
    workflow_2.fit(dataset)
    string_df = workflow_2.transform(dataset).to_ddf().compute()

    embedding_shapes = wf_ops.get_embedding_sizes(workflow_2)
    embedding_shapes_1.update(embedding_shapes)

    model = create_tf_model(["name-cat", "name-string"], [], embedding_shapes_1)
    string_df["name-cat"] = cat_df["name-cat"]

    req_dict = (
        TensorTable.from_df(string_df[["name-string", "name-cat"]].iloc[:3])
        .cpu()
        .to_dict()
    )
    predictions = model.predict(req_dict)

    triton_chain_1 = Subgraph("cat", ["name-cat"] >> TransformWorkflow(workflow))
    triton_chain_2 = Subgraph(
        "string", ["name-string"] >> TransformWorkflow(workflow_2)
    )
    triton_chain = (triton_chain_1 + triton_chain_2) >> PredictTensorflow(model)

    triton_ens = Ensemble(triton_chain, schema)
    ensemble_config, nodes_config = triton_ens.export(str(tmpdir))
    config_path = tmpdir / "executor_model" / "config.pbtxt"

    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        assert parsed.name == "executor_model"
        assert parsed.platform == "merlin_executor"
        assert hasattr(parsed, "ensemble_scheduling")

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]
    request_schema = workflow.input_schema + workflow_2.input_schema

    response = run_ensemble_on_tritonserver(
        str(tmpdir), request_schema, df, ["predictions"], ensemble_config.name
    )
    assert len(response["predictions"]) == df.shape[0]
    assert response["predictions"].tolist() == predictions["predictions"].tolist()
    assert len(response["predictions"]) == df.shape[0]


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("python", [False, True])
def test_workflow_tf_python_nvt_chain(tmpdir, dataset, engine, python):

    workflow_ops = ["name-cat", "name-string"] >> wf_ops.Categorify(cat_cache="host")
    workflow = Workflow(workflow_ops)
    workflow.fit(dataset)

    embedding_shapes = wf_ops.get_embedding_sizes(workflow)

    model = create_tf_model(["name-cat", "name-string"], [], embedding_shapes)

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]]
    response = Workflow(workflow_ops >> PredictTensorflow(model)).fit_transform(dataset)
    response = response.to_ddf().compute().reset_index(drop=True)
    assert "predictions" in response.columns
    assert response.shape[0] == df.shape[0]
