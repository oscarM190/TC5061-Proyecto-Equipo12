"""Garantiza que el Model Registry tenga un champion de P1 antes de levantar la API.

Lo ejecuta el servicio `model-seed` de docker-compose.yml antes de arrancar la API y el worker.
- Si `mibici-equipo12-p1-demanda@champion` ya existe (registro real del equipo), no hace nada.
- Si el registro está vacío (clon nuevo o runner de GitHub Actions), entrena el LightGBM de P1
  con los hiperparámetros de champion (modelo_3) sobre la muestra real de serving/seed/p1_muestra.csv y
  lo registra como champion con la etiqueta `origen=seed-demo`. No sustituye al modelo entrenado
  con la ingesta completa: solo permite levantar y probar el sistema sin descargar los datos.
"""

import os
from pathlib import Path
import sys
import time

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import pandas as pd

from mi_bici_experiment.api import MODEL_FEATURES
from mi_bici_experiment.config import OBJETIVO_COL, P1_COMBINACIONES, P1_NOMBRE_REGISTRO
from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionLGBM

ALIAS = "champion"
EXPERIMENT = "mibici-equipo12-serving"
MUESTRA = Path(os.getenv("MIBICI_SEED_CSV", "/seed/p1_muestra.csv"))
SKOPS_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
]


def wait_for_server(client: MlflowClient, retries: int = 30) -> None:
    for _ in range(retries):
        try:
            client.search_registered_models(max_results=1)
            return
        except Exception:
            time.sleep(2)
    sys.exit("El servidor de MLflow no respondió.")


def champion_actual(client: MlflowClient) -> str | None:
    """Versión con el alias champion, o None si el modelo o el alias no existen."""
    try:
        modelo = client.get_registered_model(P1_NOMBRE_REGISTRO)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    return modelo.aliases.get(ALIAS)


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    wait_for_server(client)
    version = champion_actual(client)
    if version is not None:
        print(f"{P1_NOMBRE_REGISTRO}@{ALIAS} ya existe (versión {version}); sin cambios.")
        return
    print(f"No hay {P1_NOMBRE_REGISTRO}@{ALIAS}; se registra el modelo de demostración.")

    datos = pd.read_csv(MUESTRA)
    X, y = datos[list(MODEL_FEATURES)], datos[OBJETIVO_COL]
    p1 = RegressionLGBM()
    p1.hiperparametros = P1_COMBINACIONES["modelo_3"]
    modelo = p1.construir_modelo().fit(X, y)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="P1-seed-demo", tags={"pregunta": "P1", "origen": "seed-demo"}):
        mlflow.log_params({"filas": len(datos), **p1.hiperparametros})
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(X, modelo.predict(X)),
            input_example=X.head(5),
            serialization_format="skops",
            skops_trusted_types=SKOPS_TYPES,
        )
    version = mlflow.register_model(info.model_uri, P1_NOMBRE_REGISTRO)
    client.set_model_version_tag(P1_NOMBRE_REGISTRO, version.version, "origen", "seed-demo")
    client.set_registered_model_alias(P1_NOMBRE_REGISTRO, ALIAS, version.version)
    print(f"Registrado {P1_NOMBRE_REGISTRO} v{version.version} como {ALIAS} (seed-demo).")


if __name__ == "__main__":
    main()
