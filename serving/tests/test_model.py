import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import serving.export_model as exporter
import serving.model as storage
from serving.schemas import FEATURES, StationProfile


@pytest.mark.parametrize("key", FEATURES)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_values_rejected(key, value):
    profile = dict(prop_fin_semana=0.2, log_viajes_dia=2.0, balance=0.0)
    with pytest.raises(ValidationError):
        StationProfile(**{**profile, key: value})


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    (tmp_path / "MLmodel").write_text("unit-test-model")
    (tmp_path / "model.skops").write_bytes(b"unit-test-artifact")
    metadata = {"model": {"name": "p3", "version": "1", "run_id": "original-run",
                           "uri": "models:/p3/1"}, "sha256": storage.artifact_hashes(tmp_path)}
    (tmp_path / "deployment.json").write_text(json.dumps(metadata))
    schema = SimpleNamespace(input_names=lambda: list(FEATURES),
                             inputs=[SimpleNamespace(type="DataType.double")] * 3)
    predictor = SimpleNamespace(metadata=SimpleNamespace(run_id="original-run", get_input_schema=lambda: schema))
    load = Mock(return_value=predictor)
    monkeypatch.setattr(storage.mlflow.pyfunc, "load_model", load)
    return tmp_path, load, predictor


@pytest.mark.parametrize("change", ["modified", "missing", "extra"])
def test_tampering_rejected_before_deserializing(bundle, change):
    path, load, _ = bundle
    if change == "modified":
        (path / "model.skops").write_bytes(b"modified")
    elif change == "missing":
        (path / "MLmodel").unlink()
    else:
        (path / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="SHA-256"):
        storage.load_bundle(path)
    load.assert_not_called()


def test_wrong_signature_rejected(bundle):
    path, _, predictor = bundle
    predictor.metadata.get_input_schema = lambda: None
    with pytest.raises(ValueError, match="firma"):
        storage.load_bundle(path)


def test_wrong_run_rejected(bundle):
    path, _, predictor = bundle
    predictor.metadata.run_id = "different-run"
    with pytest.raises(ValueError, match="run_id"):
        storage.load_bundle(path)


def test_alias_resolved_to_fixed_version_before_download(tmp_path, monkeypatch):
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7", run_id="run-7")
    monkeypatch.setattr(exporter, "MlflowClient", lambda: client)
    downloads = []

    def download(artifact_uri, dst_path):
        downloads.append(artifact_uri)
        (Path(dst_path) / "MLmodel").write_text("artifact")
        return dst_path

    monkeypatch.setattr(exporter.mlflow.artifacts, "download_artifacts", download)
    monkeypatch.setattr(exporter, "load_bundle", lambda path: SimpleNamespace(predict=lambda rows: [0]))
    target = tmp_path / "exported"
    metadata = exporter.export_model("p3", target, alias="champion")
    client.get_model_version_by_alias.assert_called_once_with("p3", "champion")
    assert downloads == ["models:/p3/7"]
    assert metadata["model"]["version"] == "7"
    assert metadata["sha256"] == storage.artifact_hashes(target)
    with pytest.raises(FileExistsError):
        exporter.export_model("p3", target)


def test_invalid_export_does_not_publish_partial_bundle(tmp_path, monkeypatch):
    client = Mock()
    client.get_model_version.return_value = SimpleNamespace(version="1", run_id="run-1")
    monkeypatch.setattr(exporter, "MlflowClient", lambda: client)
    monkeypatch.setattr(exporter.mlflow.artifacts, "download_artifacts", Mock(side_effect=RuntimeError("offline")))
    with pytest.raises(RuntimeError):
        exporter.export_model("p3", tmp_path / "failed", version="1")
    assert not (tmp_path / "failed").exists()
    assert list(tmp_path.iterdir()) == []
