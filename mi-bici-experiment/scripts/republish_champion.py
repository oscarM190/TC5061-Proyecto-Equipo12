"""Republish the current P1 champion with server-served artifacts.

Run from the repository root while the MLflow tracking server is running with
artifact serving enabled. The existing model remains available as an older
version after the champion alias moves to the new one.
"""

import os
import sys

import mlflow
import mlflow.artifacts
import mlflow.sklearn
from mlflow.models import Model
from mlflow.tracking import MlflowClient


MODEL_NAME = "mibici-equipo12-p1-demanda"
ALIAS = "champion"
EXPERIMENT_NAME = "mibici-equipo12-serving"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    original_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    if original_version.source.startswith("models:/m-"):
        model_id = original_version.source.removeprefix("models:/")
        if client.get_logged_model(model_id).artifact_location.startswith("mlflow-artifacts:"):
            print(
                f"Champion version {original_version.version} "
                "already has server-served artifacts."
            )
            return

    source_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    source_path = mlflow.artifacts.download_artifacts(source_uri)
    source_metadata = Model.load(source_path)
    model = mlflow.sklearn.load_model(source_path)

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    if not experiment.artifact_location.startswith("mlflow-artifacts:"):
        raise RuntimeError(
            f"{EXPERIMENT_NAME} uses {experiment.artifact_location!r}; "
            "create a fresh experiment while MLflow serves artifacts."
        )

    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name="republish-p1-champion"
    ) as run:
        mlflow.set_tags(
            {
                "source_registered_model": MODEL_NAME,
                "source_model_version": original_version.version,
                "source_run_id": original_version.run_id,
            }
        )
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=source_metadata.signature,
            serialization_format="skops",
            skops_trusted_types=source_metadata.flavors["sklearn"].get("skops_trusted_types", []),
        )

    logged_model = client.get_logged_model(info.model_uuid)
    if not logged_model.artifact_location.startswith("mlflow-artifacts:"):
        raise RuntimeError(f"Model artifacts are not served: {logged_model.artifact_location}")

    mlflow.sklearn.load_model(info.model_uri)
    version = mlflow.register_model(info.model_uri, MODEL_NAME)
    mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}")
    client.set_model_version_tag(
        MODEL_NAME, version.version, "republished_from_version", original_version.version
    )
    if client.get_model_version_by_alias(MODEL_NAME, ALIAS).version != original_version.version:
        raise RuntimeError("Champion changed while republishing; new version was not promoted.")
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)

    print(f"source_version={original_version.version}")
    print(f"experiment_id={experiment.experiment_id}")
    print(f"run_id={run.info.run_id}")
    print(f"model_uri={info.model_uri}")
    print(f"model_id={info.model_uuid}")
    print(f"artifact_location={logged_model.artifact_location}")
    print(f"champion_version={version.version}")


if __name__ == "__main__":
    main()
