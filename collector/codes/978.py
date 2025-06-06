import os

import click
import mlflow
import triton_flavor


@click.command()
@click.option(
    "--model_name",
    help="Model name",
)
@click.option(
    "--model_directory",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Model filepath",
)
@click.option(
    "--flavor",
    type=click.Choice(["triton"], case_sensitive=True),
    required=True,
    help="Model flavor",
)
def publish_to_mlflow(model_name, model_directory, flavor):
    mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    artifact_path = "triton"

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    with mlflow.start_run() as run:
        if flavor == "triton":
            triton_flavor.log_model(
                model_directory,
                artifact_path=artifact_path,
                registered_model_name=model_name,
            )
        else:

            raise Exception("Other flavor is not supported")

        print(mlflow.get_artifact_uri())


if __name__ == "__main__":
    publish_to_mlflow()
