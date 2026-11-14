import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from mi_bici_experiment.config import OBJETIVO_COL, P1_NOMBRE_REGISTRO, STATION_COL
from mi_bici_experiment.modeling import ab_test
from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionModel


def datos_actuales(n_estaciones=10, filas=30, semilla=0):
    rng = np.random.default_rng(semilla)
    n = n_estaciones * filas
    df = pd.DataFrame({
        "trips_lag_1": rng.uniform(5, 50, n),
        "trips_lag_7": rng.uniform(5, 50, n),
        "trips_roll_mean_7": rng.uniform(5, 50, n),
        "is_weekend": rng.integers(0, 2, n),
        STATION_COL: np.repeat(np.arange(1, n_estaciones + 1), filas),
        "dayofweek": rng.integers(0, 7, n),
        "month": 10,
    })
    df[OBJETIVO_COL] = 0.5 * df["trips_lag_1"] + 0.5 * df["trips_roll_mean_7"] + rng.normal(0, 0.5, n)
    return df


def test_metricas_prediccion_perfecta_y_recorte():
    assert ab_test.metricas([1, 2, 3], [1, 2, 3]) == {"mae": 0.0, "rmse": 0.0}
    assert ab_test.metricas([0, 0], [-5, 0])["mae"] == 0.0


def test_bootstrap_reproducible_y_contiene_la_media():
    dif = np.random.default_rng(1).normal(0.5, 1, 200)
    est = np.repeat(np.arange(20), 10)
    a = ab_test.intervalo_bootstrap(dif, est)
    assert a == ab_test.intervalo_bootstrap(dif, est)
    assert a[0] < dif.mean() < a[1]


def test_comparar_gana_challenger_si_es_mejor():
    actual = datos_actuales()
    y = actual[OBJETIVO_COL].to_numpy()
    malo = np.full(len(y), y.mean())
    r = ab_test.comparar(y, malo, y + 0.1, actual[STATION_COL])
    assert r["decision"] == "challenger"
    assert r["mejora_mae_pct"] > 90
    assert r["ic95_diferencia"][0] > 0
    assert r["pct_filas_gana_challenger"] > 95


def test_comparar_conserva_champion_si_son_iguales():
    actual = datos_actuales()
    y = actual[OBJETIVO_COL].to_numpy()
    r = ab_test.comparar(y, y + 1, y + 1, actual[STATION_COL])
    assert r["decision"] == "champion"
    assert r["pct_empates"] == 100.0
    assert "Se conserva el champion" in r["razon"]


def test_comparar_exige_mejora_minima():
    actual = datos_actuales()
    y = actual[OBJETIVO_COL].to_numpy()
    r = ab_test.comparar(y, y + 1.0, y + 0.9, actual[STATION_COL], min_mejora_pct=50)
    assert r["mejora_mae_pct"] > 0 and r["decision"] == "champion"


def test_cargar_actual_valida_columnas(tmp_path):
    ruta = tmp_path / "actual.parquet"
    datos_actuales().drop(columns=["month"]).to_parquet(ruta)
    with pytest.raises(ValueError, match="faltan columnas"):
        ab_test.cargar_actual(ruta)


@pytest.fixture
def registro(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("ab-pruebas", artifact_location=(tmp_path / "art").as_uri())
    monkeypatch.setattr(ab_test, "MLFLOW_EXPERIMENT", "ab-pruebas")
    monkeypatch.setattr(ab_test, "version_datos", lambda: "sha256-prueba")
    monkeypatch.setattr(ab_test, "git_info", lambda: {"git_commit": "prueba"})
    yield mlflow.MlflowClient()
    mlflow.set_tracking_uri(None)


def _registrar(modelo):
    mlflow.set_experiment("ab-pruebas")
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(modelo, artifact_path="model", serialization_format="cloudpickle")
    return str(mlflow.register_model(f"runs:/{run.info.run_id}/model", P1_NOMBRE_REGISTRO).version)


def _preparar(registro, tmp_path):
    actual = datos_actuales()
    ruta = tmp_path / "actual.parquet"
    actual.to_parquet(ruta)
    X, y = actual[RegressionModel().todas_var], actual[OBJETIVO_COL]
    v1 = _registrar(DummyRegressor().fit(X, y))     # champion: predice la media
    v2 = _registrar(LinearRegression().fit(X, y))   # challenger: aprende la relación
    registro.set_registered_model_alias(P1_NOMBRE_REGISTRO, "champion", v1)
    registro.set_registered_model_alias(P1_NOMBRE_REGISTRO, "challenger", v2)
    return ruta, v1, v2


def test_ab_sin_promover_no_mueve_alias(registro, tmp_path):
    ruta, v1, v2 = _preparar(registro, tmp_path)
    r = ab_test.evaluar_versiones(ruta)
    assert r["decision"] == "challenger" and r["promovido"] is False
    aliases = {a: str(v) for a, v in registro.get_registered_model(P1_NOMBRE_REGISTRO).aliases.items()}
    assert aliases == {"champion": v1, "challenger": v2}


def test_ab_promueve_solo_si_se_pide(registro, tmp_path):
    ruta, v1, v2 = _preparar(registro, tmp_path)
    r = ab_test.evaluar_versiones(ruta, promover=True)
    assert r["promovido"] is True
    aliases = {a: str(v) for a, v in registro.get_registered_model(P1_NOMBRE_REGISTRO).aliases.items()}
    assert aliases == {"champion": v2, "previous": v1}
    resultado = mlflow.artifacts.load_dict(
        f"runs:/{registro.get_model_version(P1_NOMBRE_REGISTRO, v1).tags['ab_test_run']}"
        "/tables/ab_test.json")
    assert resultado["decision"] == "challenger"


def test_ab_requiere_champion_y_challenger(registro, tmp_path):
    ruta = tmp_path / "actual.parquet"
    datos_actuales().to_parquet(ruta)
    with pytest.raises(ValueError, match="no tiene versión champion"):
        ab_test.evaluar_versiones(ruta)
