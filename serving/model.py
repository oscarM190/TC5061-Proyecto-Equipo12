"""Carga del artefacto exportado del Registry y comprobación de su linaje."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd

from serving.schemas import FEATURES, ModelIdentity, StationProfile


def artifact_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError("El paquete del modelo no admite enlaces simbólicos.")
        if file.is_file() and file != path / "deployment.json":
            with file.open("rb") as stream:
                hashes[file.relative_to(path).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


@dataclass
class LoadedModel:
    predictor: object
    identity: ModelIdentity

    def predict(self, profiles: list[StationProfile]) -> list[int]:
        frame = pd.DataFrame([p.model_dump() for p in profiles], columns=FEATURES).astype("float64")
        result = np.asarray(self.predictor.predict(frame))
        if result.shape != (len(profiles),) or result.dtype.kind not in "iu" or (result < 0).any():
            raise ValueError("El modelo no devolvió una etiqueta entera por perfil.")
        return result.tolist()


def load_bundle(path: Path) -> LoadedModel:
    path = path.resolve()
    metadata = json.loads((path / "deployment.json").read_text(encoding="utf-8"))
    identity = ModelIdentity.model_validate(metadata["model"])
    if identity.uri != f"models:/{identity.name}/{identity.version}":
        raise ValueError("El nombre y la versión no coinciden con la URI del Registry.")
    expected = metadata["sha256"]
    if not expected or expected != artifact_hashes(path):
        raise ValueError("Los archivos del modelo no coinciden con su manifiesto SHA-256.")
    predictor = mlflow.pyfunc.load_model(str(path))
    signature = predictor.metadata.get_input_schema()
    if signature is None or signature.input_names() != list(FEATURES):
        raise ValueError("El modelo no tiene la firma de entrada de P3.")
    if any(str(item.type) != "DataType.double" for item in signature.inputs):
        raise ValueError("Las tres variables P3 deben ser de tipo double.")
    if predictor.metadata.run_id != identity.run_id:
        raise ValueError("El run_id del modelo no coincide con el registrado al descargarlo.")
    return LoadedModel(predictor, identity)
