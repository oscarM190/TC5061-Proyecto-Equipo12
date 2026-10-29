import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from mi_bici_experiment.config import (FECHA_INICIO_COL, OBJETIVO_COL, RANDOM_STATE, STATION_COL)
from mi_bici_experiment.features import agregar_variables, construir_viajes_diarios
from mi_bici_experiment.modeling.modules.p1_regression.models import (RegressionLGBM, RegressionModel)


INICIO = "2026-01-01"
FIN = "2026-12-31"
DIAS_SINTETICOS = 150
FIN_SINTETICO = (pd.Timestamp(INICIO) + pd.Timedelta(days=DIAS_SINTETICOS - 1)).strftime("%Y-%m-%d")
LAGS = ["trips_lag_1", "trips_lag_7", "trips_roll_mean_7"]
HP_LGBM = {
    "n_estimators": 30,
    "num_leaves": 7,
    "min_child_samples": 5,
    "learning_rate": 0.1,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True}


# ----------------------------
# Datos sintéticos y fixtures
# ---------------------------
def generar_viajes(n_estaciones=4, dias=DIAS_SINTETICOS, semilla=0):
    """Un registro por viaje (estación de origen y fecha), como el Parquet de la ingesta.
    La demanda sube entre semana y varía por estación para que los modelos tengan señal."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(INICIO, periods=dias, freq="D")
    bloques = []
    for estacion in range(1, n_estaciones + 1):
        base = 6 + 4 * estacion
        lam = np.where(fechas.dayofweek < 5, base * 1.5, base * 0.7)
        conteos = rng.poisson(lam)
        dias_rep = np.repeat(fechas.to_numpy(), conteos)
        horas = pd.to_timedelta(rng.integers(0, 24 * 60, len(dias_rep)), unit="min")
        bloques.append(
            pd.DataFrame(
                {STATION_COL: np.int16(estacion), FECHA_INICIO_COL: pd.to_datetime(dias_rep) + horas}
            )
        )
    return pd.concat(bloques, ignore_index=True)


def viajes_df(filas):
    """Viajes a partir de estación y fecha ('YYYY-MM-DD HH:MM')."""
    df = pd.DataFrame(filas, columns=[STATION_COL, FECHA_INICIO_COL])
    return df.assign(**{FECHA_INICIO_COL: pd.to_datetime(df[FECHA_INICIO_COL])})


def serie_diaria(trips, estacion=1):
    fechas = pd.date_range(INICIO, periods=len(trips), freq="D")
    return pd.DataFrame({STATION_COL: estacion, "date": fechas, "trips": trips})


@pytest.fixture
def viajes():
    return generar_viajes()


@pytest.fixture
def parchar_carga(monkeypatch, viajes):
    """Hace que `cargar_viajes` use los viajes sintéticos en vez del Parquet que se usa (el real)."""
    monkeypatch.setattr(RegressionModel, "cargar_viajes", lambda self: viajes.copy())


def _preparar(clase):
    modelo = clase(INICIO, FIN_SINTETICO)
    modelo.preparar_datos()
    return modelo


@pytest.fixture
def modelo_preparado(parchar_carga):
    """RegressionModel con datos limpios y partición cronológica lista (sin entrenar)."""
    return _preparar(RegressionModel)


@pytest.fixture
def lgbm_preparado(parchar_carga):
    modelo = _preparar(RegressionLGBM)
    modelo.hiperparametros = dict(HP_LGBM)
    return modelo


# ---------------------------------------------------
# Ingeniería de variables (casos diferentes a tests)
# --------------------------------------------------
class TestViajesDiarios:
    def test_estacion_con_un_solo_viaje(self):
        v = viajes_df([(1, "2026-01-01 08:00"), (1, "2026-01-02 08:00"), (9, "2026-01-02 12:00")])
        d = construir_viajes_diarios(v, STATION_COL, FECHA_INICIO_COL)
        assert d[d[STATION_COL] == 9]["trips"].tolist() == [1] 

    def test_conserva_total_y_no_duplica(self, viajes):
        d = construir_viajes_diarios(viajes, STATION_COL, FECHA_INICIO_COL)
        assert d["trips"].sum() == len(viajes)
        assert not d.duplicated([STATION_COL, "date"]).any()


class TestAgregarVariables:
    def test_sin_fuga_de_datos(self):
        """Cambiar los viajes de un día no altera sus propias variables, solo las de días siguientes."""
        base = list(range(1, 21))
        alterada = base.copy()
        alterada[12] = 1000
        a = agregar_variables(serie_diaria(base), STATION_COL).set_index("date")
        b = agregar_variables(serie_diaria(alterada), STATION_COL).set_index("date")
        dia = pd.Timestamp(INICIO) + pd.Timedelta(days=12)
        pd.testing.assert_series_equal(a.loc[dia, LAGS], b.loc[dia, LAGS])
        assert b.loc[dia + pd.Timedelta(days=1), "trips_lag_1"] == 1000

    def test_historial_insuficiente_se_descarta(self):
        """Una estación con menos de 8 días no tiene lag_7 ni promedio móvil: sale del dataset."""
        d = pd.concat(
            [serie_diaria(list(range(1, 13)), 1), serie_diaria([3, 4, 5], 3)], ignore_index=True
        )
        r = agregar_variables(d, STATION_COL)
        assert 3 not in set(r[STATION_COL])
        assert len(r[r[STATION_COL] == 1]) == 12 - 7


# --------------------------
# Partición de datos
# --------------------------
class TestParticion:
    def test_cargar_viajes_filtra_rango_inclusivo(self, monkeypatch):
        v = viajes_df(
            [
                (1, "2025-12-31 23:59"),
                (1, "2026-01-01 00:00"),
                (1, "2026-01-31 23:59"),
                (1, "2026-02-01 00:00"),
            ]
        )
        monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: v)
        assert len(RegressionModel("2026-01-01", "2026-01-31").cargar_viajes()) == 2

    def test_cronologica_sin_traslape(self, modelo_preparado):
        m = modelo_preparado
        todo = pd.concat([m.entrenamiento, m.validacion, m.prueba])
        assert m.entrenamiento["date"].max() < m.validacion["date"].min()
        assert m.validacion["date"].max() < m.prueba["date"].min()
        assert not todo.duplicated([STATION_COL, "date"]).any()
        assert len(m.prueba) / len(todo) == pytest.approx(0.15, abs=0.05)

    def test_conjuntos_tienen_todas_las_variables(self, modelo_preparado):
        m = modelo_preparado
        for df in (m.entrenamiento, m.validacion, m.prueba):
            assert set(m.todas_var + [OBJETIVO_COL, "date"]) <= set(df.columns)

    def test_preparar_datos_es_determinista(self, parchar_carga):
        a, b = RegressionModel(INICIO, FIN), RegressionModel(INICIO, FIN)
        a.preparar_datos()
        b.preparar_datos()
        pd.testing.assert_frame_equal(a.entrenamiento, b.entrenamiento)
        pd.testing.assert_frame_equal(a.prueba, b.prueba)


# ----------------------------------------------------------
# Modelo: construcción, entrenamiento y evaluación
# ---------------------------------------------------------
class TestModelo:
    def test_baseline_es_regresion_lineal(self, modelo_preparado):
        pipe =modelo_preparado.construir_modelo()
        assert list(pipe.named_steps) == ["preprocesamiento", "modelo"]
        assert isinstance(pipe.named_steps["modelo"], LinearRegression)

    def test_lgbm_usa_semilla_e_hiperparametros(self, lgbm_preparado):
        modelo =lgbm_preparado.construir_modelo().named_steps["modelo"]
        assert isinstance(modelo, LGBMRegressor)
        assert modelo.get_params()["random_state"] == RANDOM_STATE
        assert modelo.get_params()["n_estimators"] == 30

    def test_predice_con_estacion_desconocida(self, modelo_preparado):
        m= modelo_preparado
        m.entrenar()
        nueva = m.prueba.head(5).copy()
        nueva[STATION_COL] = 999
        pred = m.modelo.predict(nueva[m.todas_var])
        assert pred.shape == (5,) and np.isfinite(pred).all()

    def test_lgbm_mejora_a_predecir_la_media(self, lgbm_preparado):
        m =lgbm_preparado
        m.entrenar()
        met =m.evaluar()
        mae_constante = (m.prueba[OBJETIVO_COL] -m.entrenamiento[OBJETIVO_COL].mean()).abs().mean()
        assert met["test_mae"] < mae_constante


class TestEvaluar:
    def test_metricas_validas(self, modelo_preparado):
        m = modelo_preparado
        m.entrenar()
        met = m.evaluar()
        assert set(met) == {"val_mae","val_rmse","test_mae","test_rmse"}
        for prefijo in ("val", "test"):
            assert 0 <= met[f"{prefijo}_mae"] <= met[f"{prefijo}_rmse"] 
        assert len(m.y_pred) == len(m.prueba)

    def test_recorta_predicciones_negativas(self, modelo_preparado):
        class Negativo:
            def predict(self, X):
                return np.full(len(X),-5.0)

        m = modelo_preparado
        m.modelo = Negativo()
        met = m.evaluar()
        assert (m.y_pred == 0).all()  # no hay viajes negativos
        assert met["test_mae"] == pytest.approx(m.prueba[OBJETIVO_COL].mean())

    def test_prediccion_perfecta_da_error_cero(self, modelo_preparado):
        class Perfecto:
            def __init__(self, *respuestas):
                self.respuestas = iter(respuestas)

            def predict(self, X):
                return next(self.respuestas)

        m = modelo_preparado
        m.modelo = Perfecto(m.validacion[OBJETIVO_COL].to_numpy(), m.prueba[OBJETIVO_COL].to_numpy())
        assert all(v == pytest.approx(0.0) for v in m.evaluar().values())


# -------------------------------------------
# Reproducibilidad del entrenamiento
# ---------------------------------------------
class TestReproducibilidad:
    @pytest.mark.parametrize(
        "clase", [RegressionModel, RegressionLGBM], ids=["baseline", "lightgbm"]
    )
    def test_misma_semilla_mismas_predicciones_y_metricas(self, parchar_carga, clase):
        """Dos ejecuciones independientes producen las mismas predicciones y métricas."""
        corridas = []
        for _ in range(2):
            m = clase(INICIO, FIN)
            m.hiperparametros = dict(HP_LGBM)
            m.preparar_datos()
            m.entrenar()
            m.evaluar()
            corridas.append((m.y_pred.copy(), dict(m.metricas)))
        print(f"\n[{clase.__name__}] corrida 1: {corridas[0][1]}")
        print(f"[{clase.__name__}] corrida 2: {corridas[1][1]}")
        print(
            f"[{clase.__name__}] diferencia máxima en predicciones: "
            f"{np.abs(corridas[0][0] - corridas[1][0]).max()}"
        )
        np.testing.assert_allclose(corridas[0][0], corridas[1][0], rtol=0, atol=1e-12)
        assert corridas[0][1] == pytest.approx(corridas[1][1])

    def test_control_negativo_otros_datos_otras_predicciones(self,lgbm_preparado):
        m = lgbm_preparado
        m.entrenar()
        m.evaluar()
        antes = m.y_pred.copy()
        m.entrenamiento = m.entrenamiento.assign(
            **{OBJETIVO_COL: m.entrenamiento[OBJETIVO_COL]* 3})
        m.entrenar()
        m.evaluar()
        assert not np.array_equal(antes, m.y_pred)
