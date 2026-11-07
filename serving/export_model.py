"""Descarga una versión real del Model Registry sin entrenar ni registrar modelos nuevos."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import mlflow
from mlflow import MlflowClient

from serving.model import artifact_hashes, load_bundle
from serving.schemas import ModelIdentity, StationProfile


def export_model(name: str, destination: Path, version: str | None = None,
                 alias: str = "champion") -> dict:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Ya existe {destination}; usa otra carpeta para otra versión.")
    client = MlflowClient()
    selected = (client.get_model_version(name, version) if version
                else client.get_model_version_by_alias(name, alias))
    if not selected.run_id:
        raise ValueError("La versión del Registry no tiene run_id de origen.")
    # Resolver el alias una sola vez impide que un cambio de champion mezcle versiones.
    uri = f"models:/{name}/{selected.version}"
    identity = ModelIdentity(name=name, version=str(selected.version),
                             run_id=selected.run_id, uri=uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".export-") as temporary:
        download_dir = Path(temporary) / "download"
        download_dir.mkdir()
        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(download_dir)))
        bundle = Path(temporary) / "bundle"
        shutil.copytree(downloaded, bundle)
        metadata = {
            "model": identity.model_dump(),
            "requested_alias": None if version else alias,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "sha256": artifact_hashes(bundle),
        }
        (bundle / "deployment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        loaded = load_bundle(bundle)
        loaded.predict([StationProfile(prop_fin_semana=0.2, log_viajes_dia=3.0, balance=0.0)])
        bundle.rename(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="mibici-equipo12-p3-grupos")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--version")
    selector.add_argument("--alias", default="champion")
    parser.add_argument("--output", type=Path, default=Path("artifacts/p3"))
    args = parser.parse_args()
    print(json.dumps(export_model(args.name, args.output, args.version, args.alias), indent=2))


if __name__ == "__main__":
    main()
