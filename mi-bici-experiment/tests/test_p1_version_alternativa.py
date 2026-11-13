"""Versión alternativa de P1: reentrenamiento registrado como nueva versión (challenger)."""

import mlflow
import pytest

from mi_bici_experiment.config import P1_NOMBRE_REGISTRO
from mi_bici_experiment.modeling.modules.p1_regression import models as p1
from mi_bici_experiment.modeling.run import _argumentos_p1, main
from test_p1 import FIN_SINTETICO, HP_LGBM, INICIO, generar_viajes

# LightGBM pequeño y determinista para que las pruebas sean rápidas.
RECETA = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5}
FIJOS = {k: v for k, v in HP_LGBM.items() if k not in RECETA}


@pytest.fixture
def registro(tmp_path, monkeypatch):
    """Model Registry temporal, viajes sintéticos y sin depender del manifiesto ni de Git."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("p1-pruebas", artifact_location=(tmp_path / "artefactos").as_uri())
    monkeypatch.setattr(p1, "MLFLOW_EXPERIMENT", "p1-pruebas")
    manifiesto = tmp_path / "data_manifest.json"
    manifiesto.write_text("{}")
    monkeypatch.setattr(p1, "MANIFIESTO", manifiesto)
    monkeypatch.setattr(p1, "version_datos", lambda: "sha256-prueba")
    monkeypatch.setattr(p1, "git_info", lambda: {"git_commit": "prueba", "git_branch": "prueba",
                                                 "git_dirty": "false"})
    viajes = generar_viajes()
    monkeypatch.setattr(p1.RegressionModel, "cargar_viajes", lambda self: viajes.copy())
    # Hiperparámetros fijos (hilos y determinismo) en todas las versiones, sin registrarlos.
    construir = p1.RegressionLGBM.construir_modelo

    def construir_determinista(self):
        pipeline = construir(self)
        pipeline.named_steps["modelo"].set_params(**FIJOS)
        return pipeline

    monkeypatch.setattr(p1.RegressionLGBM, "construir_modelo", construir_determinista)
    yield mlflow.MlflowClient()
    mlflow.set_tracking_uri(None)


def _champion():
    """Versión champion como la deja la búsqueda de P1 (solo entrenamiento, receta RECETA)."""
    modelo = p1.RegressionLGBM(INICIO, FIN_SINTETICO)
    modelo.hiperparametros = dict(RECETA)
    modelo.preparar_datos()
    modelo.entrenar()
    modelo.evaluar()
    mlflow.set_experiment("p1-pruebas")
    with mlflow.start_run() as run:
        modelo.registrar_corrida()
    p1.promover_a_produccion(run.info.run_id)
    return modelo


def test_hiperparametros_de_corrida_recupera_tipos():
    params = {"n_estimators": "400", "learning_rate": "0.05", "num_leaves": "31", "modelo": "x"}
    assert p1.hiperparametros_de_corrida(params) == {
        "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31}


def test_reentrenar_registra_challenger_sin_mover_champion(registro):
    champion = _champion()

    version, base = p1.reentrenar(INICIO, FIN_SINTETICO)

    assert (version, base) == ("2", "1")
    modelo = registro.get_registered_model(P1_NOMBRE_REGISTRO)
    aliases = {a: str(v) for a, v in modelo.aliases.items()}
    assert aliases == {"champion": "1", "challenger": "2"}
    nueva = registro.get_model_version(P1_NOMBRE_REGISTRO, version)
    assert nueva.tags["version_base"] == "1"
    run = registro.get_run(nueva.run_id)
    assert p1.hiperparametros_de_corrida(run.data.params) == RECETA
    assert run.data.tags["stage_of_work"] == "retrain"
    # Entrena con entrenamiento + validación: llega justo antes de la prueba de champion.
    assert run.data.params["fin_entrenamiento"] < str(champion.prueba["date"].min().date())
    assert run.data.params["fin_entrenamiento"] == str(champion.validacion["date"].max().date())
    # La validación ya está en el entrenamiento: solo se registran métricas de prueba.
    assert set(run.data.metrics) == {"test_mae", "test_rmse"}
    assert run.data.params["incluir_validacion"] == "true"
    comparacion = mlflow.artifacts.load_dict(
        f"runs:/{nueva.run_id}/tables/comparacion_con_champion.json")
    assert comparacion["misma_prueba"] is True
    assert comparacion["champion"]["metricas"]["test_mae"] == pytest.approx(
        champion.metricas["test_mae"])


def test_reentrenar_con_otros_hiperparametros(registro):
    _champion()
    version, _ = p1.reentrenar(INICIO, FIN_SINTETICO, n_estimators=60)
    run = registro.get_run(registro.get_model_version(P1_NOMBRE_REGISTRO, version).run_id)
    assert p1.hiperparametros_de_corrida(run.data.params) == {**RECETA, "n_estimators": 60}


def test_reentrenar_requiere_champion_y_una_version_distinta(registro):
    with pytest.raises(ValueError, match="no tiene versión champion"):
        p1.reentrenar(INICIO, FIN_SINTETICO)
    _champion()
    with pytest.raises(ValueError, match="no sería una versión distinta"):
        p1.reentrenar(INICIO, FIN_SINTETICO, incluir_validacion=False)
    with pytest.raises(ValueError, match="champion"):
        p1.reentrenar(INICIO, FIN_SINTETICO, alias="champion")


def test_argumentos_de_retrain():
    a = _argumentos_p1([])
    assert (a.incluir_validacion, a.alias) == ("true", "challenger")
    a = _argumentos_p1(["--incluir_validacion", "false", "--n_estimators", "500"])
    assert (a.incluir_validacion, a.n_estimators) == ("false", 500)


def test_mlproject_retrain_por_defecto_es_p1_challenger():
    import yaml

    from mi_bici_experiment.config import PROJ_ROOT

    mlproject = yaml.safe_load((PROJ_ROOT.parent / "MLproject").read_text())
    parametros = mlproject["entry_points"]["retrain"]["parameters"]
    assert parametros["question"]["default"] == "p1"
    assert parametros["alias"]["default"] == "challenger"


def test_retrain_solo_para_p1():
    with pytest.raises(SystemExit, match="solo está implementado para p1"):
        main(["retrain", "--question", "p3"])
