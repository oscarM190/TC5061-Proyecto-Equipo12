from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from mi_bici_experiment.modeling.modules.p2_classification import models


@pytest.fixture
def dataset():
    template = models.ClassificationModel(pd.DataFrame())
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(120, len(template.numeric_features))),
        columns=template.numeric_features,
    )
    frame["Estacion_Id"] = np.tile([1, 2, 3], 40)
    frame["Fecha"] = np.repeat(pd.date_range("2026-01-01", periods=40), 3)
    frame["Fecha_Objetivo"] = frame["Fecha"] + pd.Timedelta(days=1)
    frame["Riesgo_T1"] = np.arange(len(frame)) % 3
    return frame.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture
def model(dataset):
    instance = models.ClassificationModel(dataset)
    # Keep real Random Forest training small in unit tests.
    instance.rf_default_params.update(n_estimators=8, n_jobs=1)
    return instance


@pytest.fixture
def tracking(monkeypatch, tmp_path):
    mock = MagicMock()
    mock.start_run.return_value.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-sintetico")
    )
    mock.sklearn.log_model.return_value = SimpleNamespace(model_uri="models:/modelo-sintetico")
    monkeypatch.setattr(models, "mlflow", mock)
    monkeypatch.setattr(models, "version_datos", lambda: "sha256-sintetico")
    monkeypatch.setattr(models, "git_info", lambda: {"git_commit": "commit-sintetico"})
    monkeypatch.setattr(models, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(
        models, "matriz_confucion_regresion_multiclass_clasificacion",
        MagicMock(return_value=tmp_path / "matriz.png"),
    )
    return mock


def test_particion_agrupa_fechas_y_purga_objetivos(model, dataset):
    original = dataset.copy(deep=True)
    train, val, test = model.separar_datos()
    # 28/6/6 dates before purging one day at each internal boundary.
    assert [part.Fecha.nunique() for part in (train, val, test)] == [27, 5, 6]
    for part in (train, val, test):
        assert part.Fecha.is_monotonic_increasing
        assert part.groupby("Fecha").Estacion_Id.nunique().eq(3).all()
    assert train.Fecha.max() < val.Fecha.min() < test.Fecha.min()
    assert train.Fecha_Objetivo.max() < val.Fecha.min()
    assert val.Fecha_Objetivo.max() < test.Fecha.min()
    assert model.split_params["n_purgadas_train"] == 3
    assert model.split_params["n_purgadas_validacion"] == 3
    assert_frame_equal(dataset, original)


def test_purga_fecha_objetivo_real_aunque_salte_varios_dias(model):
    mask = model.dataset_ml.Fecha.eq("2026-01-20") & model.dataset_ml.Estacion_Id.eq(1)
    model.dataset_ml.loc[mask, "Fecha_Objetivo"] = pd.Timestamp("2026-02-01")
    train, _, _ = model.separar_datos()
    assert not (train.Fecha.eq("2026-01-20") & train.Estacion_Id.eq(1)).any()


@pytest.mark.parametrize("case, message", [
    ("sin_objetivo", "Falta Fecha_Objetivo"),
    ("nulo", "nulos"),
    ("objetivo_pasado", "posterior"),
    ("pocas_fechas", "suficientes fechas"),
    ("train_vacio", "vacío"),
])
def test_particion_rechaza_datos_invalidos(model, case, message):
    if case == "sin_objetivo":
        model.dataset_ml = model.dataset_ml.drop(columns="Fecha_Objetivo")
    elif case == "nulo":
        model.dataset_ml.loc[0, "Fecha"] = pd.NaT
    elif case == "objetivo_pasado":
        model.dataset_ml.loc[0, "Fecha_Objetivo"] = model.dataset_ml.loc[0, "Fecha"]
    elif case == "pocas_fechas":
        model.dataset_ml = model.dataset_ml.head(1)
    else:
        model.dataset_ml["Fecha_Objetivo"] = pd.Timestamp("2026-03-01")
    with pytest.raises(ValueError, match=message):
        model.separar_datos()


@pytest.mark.parametrize("family", ["lr", "rf"])
def test_entrenamiento_real_y_registro_simulado(model, tracking, family):
    train, _, _ = model.separar_datos()
    if family == "lr":
        fitted = model.entrenar_regresion_logistica()
        untouched = model.pipeline_rf
    else:
        fitted = model.entrenar_random_forest({"n_estimators": 8, "n_jobs": 1}, nested=True)
        untouched = model.pipeline_lr
    scaler = fitted.named_steps["preprocesamiento"].named_transformers_["numericas"]
    np.testing.assert_allclose(scaler.mean_, train[model.numeric_features].mean())
    assert not hasattr(untouched.named_steps["preprocesamiento"], "transformers_")
    prediction = fitted.predict(train[model.features].head())
    assert len(prediction) == 5
    assert set(prediction) <= {0, 1, 2}

    logged = {}
    for call in tracking.log_metrics.call_args_list:
        logged.update(call.args[0])
    expected = {
        f"{split}{suffix}_{metric}"
        for split in ("val", "test")
        for suffix in ("", "_clase_0", "_clase_1", "_clase_2")
        for metric in ("precision", "recall", "f1-score")
    }
    assert set(logged) == expected
    assert all(0 <= value <= 1 for value in logged.values())
    params = tracking.log_params.call_args.args[0]
    assert params["n_train"] == len(train)
    assert params["version_datos"] == "sha256-sintetico"
    assert params["clasificador__random_state"] == models.RANDOM_STATE
    assert tracking.log_artifact.call_count == 3
    saved = tracking.sklearn.log_model.call_args
    assert saved.args[0] is fitted
    assert model.last_model_uri == "models:/modelo-sintetico"
    assert model.last_run_id == "run-sintetico"
    if family == "rf":
        assert saved.kwargs["registered_model_name"] is None
        assert params["clasificador__n_estimators"] == 8
    else:
        assert saved.kwargs["registered_model_name"] == "regresion-lineal-multinomial-clasificacion"


@pytest.mark.parametrize("family", ["lr", "rf"])
def test_reproducibilidad_entrenamiento(dataset, tracking, family):
    """Mismos datos, parámetros y semilla producen los mismos resultados en este entorno."""
    instances = [models.ClassificationModel(dataset.copy(deep=True)) for _ in range(2)]
    partitions = [instance.separar_datos() for instance in instances]
    for first, second in zip(*partitions):
        assert_frame_equal(first, second)

    fitted_models = []
    logged_metrics = []
    for instance in instances:
        tracking.log_metrics.reset_mock()
        if family == "lr":
            fitted = instance.entrenar_regresion_logistica()
        else:
            fitted = instance.entrenar_random_forest({
                "n_estimators": 8, "n_jobs": 1, "random_state": models.RANDOM_STATE,
            })
        assert fitted.named_steps["clasificador"].random_state == models.RANDOM_STATE
        fitted_models.append(fitted)
        metrics = {}
        for call in tracking.log_metrics.call_args_list:
            metrics.update(call.args[0])
        logged_metrics.append(metrics)

    first, second = fitted_models
    assert first is not second
    assert first.named_steps["clasificador"] is not second.named_steps["clasificador"]
    assert first.named_steps["preprocesamiento"] is not second.named_steps["preprocesamiento"]
    for evaluation in partitions[0][1:]:  # Validation and test.
        X = evaluation[instances[0].features]
        np.testing.assert_array_equal(first.classes_, second.classes_)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        np.testing.assert_allclose(
            first.predict_proba(X), second.predict_proba(X), rtol=1e-12, atol=1e-12,
        )
    assert len(logged_metrics[0]) == 24
    assert logged_metrics[0] == pytest.approx(logged_metrics[1], rel=1e-12, abs=1e-12)


def test_candidatos_no_heredan_hiperparametros(model, monkeypatch):
    monkeypatch.setattr(model, "_entrenar_pipeline", MagicMock())
    original_depth = model.rf_default_params["max_depth"]
    model.entrenar_random_forest({"max_depth": 3})
    model.entrenar_random_forest({"n_estimators": 12})
    classifier = model.pipeline_rf.named_steps["clasificador"]
    assert classifier.max_depth == original_depth
    assert classifier.n_estimators == 12


def test_busqueda_elige_validacion_y_registra_el_ganador(model, tracking, monkeypatch):
    scores = {"primero": .4, "ganador": .8, "ultimo": .5}

    def candidate(params, nombre, nested):
        assert nested is True
        model.last_run_id = f"run-{nombre}"
        model.last_model_uri = f"models:/modelo-{nombre}"
        # The best test score deliberately belongs to a different candidate.
        model.metricas = {"val_f1-score": scores[nombre], "test_f1-score": 1 - scores[nombre]}

    train = MagicMock(side_effect=candidate)
    register = MagicMock(return_value=SimpleNamespace(version="7"))
    monkeypatch.setattr(model, "entrenar_random_forest", train)
    monkeypatch.setattr(model, "registrar_mejor_modelo", register)
    best = model.buscar_hiperparametros({name: {} for name in scores})
    assert train.call_count == 3
    assert best["run_id"] == "run-ganador"
    assert best["model_uri"] == "models:/modelo-ganador"
    register.assert_called_once_with(best)
    tracking.set_tag.assert_called_with("best_run_id", "run-ganador")


def test_busqueda_vacia_no_inicia_runs(model, tracking):
    with pytest.raises(ValueError, match="No hay configuraciones"):
        model.buscar_hiperparametros({})
    tracking.start_run.assert_not_called()


def test_registro_conserva_linaje_y_asigna_champion(model, tracking):
    client = tracking.MlflowClient.return_value
    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(
        tags={"git_commit": "commit-original", "pregunta": "P2"},
        params={"version_datos": "sha256-original"},
    ))
    tracking.register_model.return_value = SimpleNamespace(version="4")
    best = {"run_id": "winner", "model_uri": "models:/winner",
            "combinacion": "modelo_2", "val_f1_score": .7}
    model.registrar_mejor_modelo(best)
    tracking.register_model.assert_called_once_with(
        model_uri="models:/winner", name=models.P2_NOMBRE_REGISTRO
    )
    client.set_model_version_tag.assert_any_call(
        models.P2_NOMBRE_REGISTRO, "4", "data_version", "sha256-original"
    )
    client.set_model_version_tag.assert_any_call(
        models.P2_NOMBRE_REGISTRO, "4", "git_commit", "commit-original"
    )
    client.set_registered_model_alias.assert_called_once_with(
        models.P2_NOMBRE_REGISTRO, "champion", "4"
    )


def test_fallo_registro_no_modifica_champion(model, tracking):
    tracking.register_model.side_effect = RuntimeError("registro no disponible")
    with pytest.raises(RuntimeError, match="registro no disponible"):
        model.registrar_mejor_modelo({"run_id": "winner", "model_uri": "models:/winner"})
    tracking.MlflowClient.return_value.set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("args", [[], ["--n-estimators", "17", "--max-depth", "5"]])
def test_cli_busqueda_o_combinacion_personalizada(monkeypatch, args):
    execute = MagicMock()
    monkeypatch.setattr(models.ClassificationModel, "ejecutar", execute)
    result = CliRunner().invoke(models.app, args)
    assert result.exit_code == 0, result.output
    if not args:
        execute.assert_called_once_with()
    else:
        execute.assert_called_once_with({
            **models.P2_COMBINACIONES["modelo_3"], "n_estimators": 17, "max_depth": 5,
        })


@pytest.mark.parametrize("args", [
    ["--n-estimators", "0"],
    ["--max-features", "incorrecto"],
    ["--no-bootstrap", "--max-samples", "0.5"],
])
def test_cli_invalida_no_entrena(monkeypatch, args):
    execute = MagicMock()
    monkeypatch.setattr(models.ClassificationModel, "ejecutar", execute)
    result = CliRunner().invoke(models.app, args)
    assert result.exit_code != 0
    execute.assert_not_called()


@pytest.mark.parametrize("options, expected", [
    ({"solo_baseline": True}, "lr"),
    ({"hiperparametros": {"n_estimators": 8}}, "rf"),
    ({}, "search"),
])
def test_ejecutar_respeta_el_modo_solicitado(monkeypatch, tmp_path, dataset, options, expected):
    monkeypatch.setattr(models.pd, "read_parquet", MagicMock(return_value=dataset.copy()))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", MagicMock())
    monkeypatch.setattr(models, "P2_DATASET_PARQUET", tmp_path / "dataset.parquet")
    for function in (
        "agregar_balance_neto_y_normalizado",
        "clasificacion_variable_objetivo_y_caracterizticas",
        "creacion_variables_memoria_historica",
    ):
        monkeypatch.setattr(models, function, MagicMock(return_value=dataset.copy()))
    lr = MagicMock(return_value="lr")
    rf = MagicMock(return_value="rf")
    search = MagicMock(return_value="search")
    monkeypatch.setattr(models.ClassificationModel, "entrenar_regresion_logistica", lr)
    monkeypatch.setattr(models.ClassificationModel, "entrenar_random_forest", rf)
    monkeypatch.setattr(models.ClassificationModel, "buscar_hiperparametros", search)
    assert models.ClassificationModel.ejecutar(**options) == expected
    assert lr.call_count == (1 if expected in ("lr", "search") else 0)
    assert rf.call_count == (1 if expected == "rf" else 0)
    assert search.call_count == (1 if expected == "search" else 0)
    if expected == "rf":
        rf.assert_called_once_with(options["hiperparametros"], nombre="personalizada")
