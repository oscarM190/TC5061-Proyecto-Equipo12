"""Pruebas de P3 (clustering de estaciones) con datos sintéticos."""

import json
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from mi_bici_experiment.config import (
    DESTINO_COL,
    EQUIPO,
    FECHA_INICIO,
    FECHA_FIN,
    FECHA_FIN_COL,
    FECHA_INICIO_COL,
    MI_BICI_CSVS,
    MLFLOW_EXPERIMENT,
    P3_FIN_ENTRENAMIENTO,
    P3_INICIO_ENTRENAMIENTO,
    P3_MIN_PROP_GRUPO,
    STATION_COL,
)
from mi_bici_experiment.features import (
    columnas_perfil,
    filtrar_viajes_validos,
    perfil_estaciones,
    resumen_uso,
)
from mi_bici_experiment.modeling.modules.p3_clustering import models as p3
from mi_bici_experiment.modeling.run import _argumentos_p3
from mi_bici_experiment.tracking import version_datos_manifiesto
from mi_bici_experiment import validacion_p3


def viajes_sinteticos(n=3000, seed=0):
    """Viajes entre 4 estaciones durante dos meses, con fechas truncadas al día como la ingesta."""
    rng = np.random.default_rng(seed)
    inicio = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 59, n), "D")
    return pd.DataFrame(
        {
            STATION_COL: rng.integers(1, 5, n),
            DESTINO_COL: rng.integers(1, 5, n),
            FECHA_INICIO_COL: inicio,
            FECHA_FIN_COL: inicio + pd.to_timedelta(rng.integers(0, 2, n), "D"),
        }
    )


def perfil_sintetico(n=90, seed=0):
    """Tres patrones de uso bien separados, con el formato de `perfil_estaciones`."""
    rng = np.random.default_rng(seed)
    centros = np.array([[0.15, 1.5, -0.2], [0.20, 3.5, 0.05], [0.40, 3.0, -0.05]])
    grupo = np.repeat(np.arange(3), n // 3)
    valores = centros[grupo] + rng.normal(0, [0.02, 0.2, 0.03], (len(grupo), 3))
    perfil = pd.DataFrame(valores, columns=["prop_fin_semana", "log_viajes_dia", "balance"])
    return perfil.assign(estacion_id=np.arange(len(perfil)))[
        ["estacion_id", "prop_fin_semana", "log_viajes_dia", "balance"]
    ]


def test_filtra_nulos_y_fechas_invertidas():
    t = pd.Timestamp("2026-01-05")
    viajes = pd.DataFrame(
        {
            STATION_COL: [1, 1, None],
            DESTINO_COL: [2, 2, 2],
            FECHA_INICIO_COL: [t, t, t],
            FECHA_FIN_COL: [t, t - pd.Timedelta(days=1), t],
        }
    )
    assert len(filtrar_viajes_validos(viajes)) == 1


def test_resumen_en_rango():
    resumen = resumen_uso(viajes_sinteticos())
    assert resumen["balance"].between(-1, 1).all()
    assert resumen["prop_fin_semana"].between(0, 1).all()


def test_perfil_no_usa_datos_posteriores():
    """Cambiar los viajes posteriores al fin del periodo no altera el perfil."""
    viajes = viajes_sinteticos()
    fin = "2026-01-31"
    base = perfil_estaciones(viajes, "2026-01-01", fin, min_viajes=1)

    alterados = viajes.copy()
    futuros = alterados[FECHA_INICIO_COL] > pd.Timestamp(fin)
    alterados.loc[futuros, STATION_COL] = 1
    alterados.loc[futuros, DESTINO_COL] = 2
    pd.testing.assert_frame_equal(
        base, perfil_estaciones(alterados, "2026-01-01", fin, min_viajes=1)
    )


def test_particion_temporal_sin_traslape():
    """Ajuste de validación < validación <= fin del entrenamiento < prueba <= FECHA_FIN."""
    fechas = [
        P3_INICIO_ENTRENAMIENTO,
        p3.FIN_AJUSTE_VALIDACION,
        p3.INICIO_VALIDACION,
        P3_FIN_ENTRENAMIENTO,
        p3.INICIO_PRUEBA,
        FECHA_FIN,
    ]
    assert [pd.Timestamp(f) for f in fechas] == sorted(pd.Timestamp(f) for f in fechas)
    assert pd.Timestamp(p3.INICIO_VALIDACION) - pd.Timestamp(p3.FIN_AJUSTE_VALIDACION) == pd.Timedelta(days=1)
    assert pd.Timestamp(p3.INICIO_PRUEBA) - pd.Timestamp(P3_FIN_ENTRENAMIENTO) == pd.Timedelta(days=1)


@pytest.mark.parametrize(
    "model,kw",
    [("kmeans", {}), ("kmeans", {"pca": 2}), ("gmm", {"cov": "spherical"}), ("aglomerativo", {})],
)
def test_modelos_recuperan_grupos_y_asignan_nuevas(model, kw):
    perfil = perfil_sintetico()
    cols = columnas_perfil(perfil)
    pipe = p3.construir(model, 3, **kw).fit(perfil[cols])
    grupos = p3.etiquetas(pipe, perfil[cols])
    assert len(np.unique(grupos)) == 3
    # Mismo perfil en el periodo posterior: la asignación coincide con los grupos de ajuste
    assert p3.ari_estabilidad(pipe, perfil, perfil, cols) == pytest.approx(1.0)
    assert set(pipe.predict(perfil[cols].head(5))) <= set(grupos)


def test_modelo_desconocido():
    with pytest.raises(ValueError):
        p3.construir("dbscan", 3)


def test_version_datos_ignora_parquet(tmp_path):
    """La versión depende de los CSV fuente, no del Parquet (cuyo binario no es estable)."""
    def manifiesto(hash_parquet):
        return {
            "files": [
                {"path": "data/raw/a.csv", "role": "source", "sha256": "aaa"},
                {"path": "data/raw/b.csv", "role": "source", "sha256": "bbb"},
                {"path": "data/interim/x.parquet", "role": "output", "sha256": hash_parquet},
            ]
        }

    rutas = []
    for i, h in enumerate(("111", "222")):
        ruta = tmp_path / f"m{i}.json"
        ruta.write_text(json.dumps(manifiesto(h)))
        rutas.append(ruta)
    assert version_datos_manifiesto(rutas[0]) == version_datos_manifiesto(rutas[1])


def test_argumentos_por_defecto_desde_config():
    a = _argumentos_p3([])
    assert a.inicio == P3_INICIO_ENTRENAMIENTO
    a = _argumentos_p3(["--k", "4", "--pca", "2", "--n_init", "10"])
    assert (a.k, a.pca, a.n_init) == (4, 2, 10)


def test_experimento_incluye_equipo():
    assert MLFLOW_EXPERIMENT == f"mibici-equipo{EQUIPO}"


def test_rango_de_fechas_coherente():
    inicio, fin = pd.Timestamp(FECHA_INICIO), pd.Timestamp(FECHA_FIN)
    assert inicio < pd.Timestamp(P3_FIN_ENTRENAMIENTO) < fin
    assert pd.Timestamp(P3_INICIO_ENTRENAMIENTO) < pd.Timestamp(P3_FIN_ENTRENAMIENTO)


def test_csvs_cubren_el_rango():
    meses = {u.rsplit("_", 2)[-2] + "-" + u.rsplit("_", 1)[-1][:2] for u in MI_BICI_CSVS if "datos_abiertos" in u}
    # Los CSV deben cubrir el rango de P1/P2 (FECHA_INICIO) y el de P3 (P3_INICIO_ENTRENAMIENTO)
    inicio = min(pd.Timestamp(FECHA_INICIO), pd.Timestamp(P3_INICIO_ENTRENAMIENTO))
    esperados = {p.strftime("%Y-%m") for p in pd.period_range(inicio, FECHA_FIN, freq="M")}
    assert esperados <= meses


# --- Reproducibilidad del entrenamiento ---------------------------------------------------------

def datos_sinteticos(seed=0):
    """Objeto con la forma de `DatosP3`: perfiles de ajuste y de un periodo posterior con ruido."""
    perfil = perfil_sintetico(seed=seed)
    cols = columnas_perfil(perfil)
    rng = np.random.default_rng(seed + 1)
    posterior = perfil.copy()
    posterior[cols] += rng.normal(0, [0.01, 0.1, 0.01], (len(perfil), 3))
    return SimpleNamespace(inicio="2026-01-01", train=perfil, ajuste_val=perfil, val=posterior,
                           test=posterior, cols=cols)


CONFIGURACIONES = [
    ("kmeans", {}),
    ("kmeans", {"pca": 2}),
    ("gmm", {"cov": "spherical"}),
    ("aglomerativo", {}),
]


@pytest.mark.parametrize("model,kw", CONFIGURACIONES, ids=["kmeans", "kmeans_pca2", "gmm", "aglomerativo"])
def test_misma_semilla_mismos_grupos_y_metricas(model, kw):
    """Dos ajustes con la misma semilla dan las mismas etiquetas, predicciones y métricas."""
    datos = datos_sinteticos()
    X = datos.train[datos.cols]

    def ajustar():
        pipe = p3.construir(model, 3, seed=7, **kw).fit(X)
        pipe_val = p3.construir(model, 3, seed=7, **kw).fit(datos.ajuste_val[datos.cols])
        return pipe, p3.metricas(pipe, pipe_val, datos)

    pipe_a, metricas_a = ajustar()
    pipe_b, metricas_b = ajustar()
    np.testing.assert_array_equal(p3.etiquetas(pipe_a, X), p3.etiquetas(pipe_b, X))
    np.testing.assert_array_equal(pipe_a.predict(datos.test[datos.cols]),
                                  pipe_b.predict(datos.test[datos.cols]))
    assert metricas_a == metricas_b


def test_grupos_no_dependen_de_la_semilla():
    """Con grupos bien separados, otra semilla solo permuta los números de grupo (ARI = 1)."""
    datos = datos_sinteticos()
    X = datos.train[datos.cols]
    grupos = [p3.etiquetas(p3.construir("kmeans", 3, pca=2, seed=s).fit(X), X) for s in (0, 1, 42)]
    assert adjusted_rand_score(grupos[0], grupos[1]) == pytest.approx(1.0)
    assert adjusted_rand_score(grupos[0], grupos[2]) == pytest.approx(1.0)


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """Servidor de tracking local y desechable; la versión de datos no depende del manifiesto."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("p3-pruebas", artifact_location=(tmp_path / "artefactos").as_uri())
    mlflow.set_experiment("p3-pruebas")
    monkeypatch.setattr(p3, "version_datos_manifiesto", lambda: "sha256-prueba")
    yield mlflow.MlflowClient()
    mlflow.set_tracking_uri(None)


def test_entrenar_reproducible_en_mlflow(mlflow_temporal):
    """`entrenar` (la función que usa `mlflow run`) registra las mismas métricas y parámetros."""
    datos = datos_sinteticos()
    corridas = [
        mlflow_temporal.get_run(p3.entrenar(datos, "kmeans", 3, pca=2, seed=7, artefactos=False)[0])
        for _ in range(2)
    ]
    a, b = (c.data for c in corridas)
    assert a.metrics == b.metrics
    assert a.params == b.params
    assert a.params["random_state"] == "7"
    assert a.tags["pregunta"] == "P3" and a.tags["equipo"] == EQUIPO


def test_busqueda_elige_k_real_y_respeta_tamano_minimo(mlflow_temporal):
    """La búsqueda con nested runs recupera los 3 grupos sintéticos y lo registra en el padre."""
    datos = datos_sinteticos()
    padre, k, pca = p3.buscar(datos, "kmeans", [2, 3, 4], [0])
    assert (k, pca) == (3, 0)
    hijas = mlflow_temporal.search_runs(
        [mlflow_temporal.get_run(padre).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{padre}'",
    )
    assert len(hijas) == 3
    assert all(h.data.metrics["train_prop_min_grupo"] >= P3_MIN_PROP_GRUPO
               for h in hijas if h.data.params["k"] == "3")


# --- Casos límite --------------------------------------------------------------------------------

def test_metricas_en_rango():
    datos = datos_sinteticos()
    pipe = p3.construir("kmeans", 3).fit(datos.train[datos.cols])
    m = p3.metricas(pipe, pipe, datos)
    assert -1 <= m["train_silhouette_espacio_comun"] <= 1
    assert m["train_davies_bouldin_espacio_comun"] >= 0
    assert m["train_prop_min_grupo"] == pytest.approx(m["train_tamano_min_grupo"] / len(datos.train))
    assert -1 <= m["test_ari_estabilidad"] <= 1


def test_periodo_sin_viajes_da_perfil_vacio():
    perfil = perfil_estaciones(viajes_sinteticos(), "2027-01-01", "2027-01-31", min_viajes=1)
    assert perfil.empty
    assert list(perfil.columns) == ["estacion_id", "prop_fin_semana", "log_viajes_dia", "balance"]


def test_min_viajes_excluye_estaciones_poco_usadas():
    viajes = viajes_sinteticos()
    salidas = viajes[STATION_COL].value_counts()
    umbral = int(salidas.median()) + 1
    perfil = perfil_estaciones(viajes, "2026-01-01", "2026-03-01", min_viajes=umbral)
    assert set(perfil["estacion_id"]) == set(salidas[salidas >= umbral].index)
    assert perfil_estaciones(viajes, "2026-01-01", "2026-03-01", min_viajes=10**6).empty


def test_valores_extremos_del_resumen():
    """Solo salidas en sábado: prop_fin_semana = 1 y balance = -1; quien solo recibe no se perfila."""
    sabado = pd.Timestamp("2026-01-03")
    viajes = pd.DataFrame({STATION_COL: [7, 7], DESTINO_COL: [8, 8],
                           FECHA_INICIO_COL: [sabado] * 2, FECHA_FIN_COL: [sabado] * 2})
    resumen = resumen_uso(viajes)
    assert list(resumen.index) == [7]
    assert resumen.loc[7, "prop_fin_semana"] == 1
    assert resumen.loc[7, "balance"] == -1
    assert resumen.loc[7, "log_viajes_dia"] == pytest.approx(np.log1p(2))


def test_ari_sin_estaciones_comunes_falla():
    """Si ninguna estación sigue activa en el periodo posterior, la estabilidad no es medible."""
    datos = datos_sinteticos()
    pipe = p3.construir("kmeans", 3).fit(datos.train[datos.cols])
    otras = datos.test.assign(estacion_id=datos.test["estacion_id"] + 10_000)
    with pytest.raises(ValueError):
        p3.ari_estabilidad(pipe, datos.train, otras, datos.cols)


def test_mas_grupos_que_estaciones_falla():
    perfil = perfil_sintetico(n=9)
    with pytest.raises(ValueError):
        p3.construir("kmeans", 10).fit(perfil[columnas_perfil(perfil)])


def test_perfil_con_nulos_falla():
    """El pipeline no imputa: un perfil con NaN debe fallar en lugar de agrupar en silencio."""
    perfil = perfil_sintetico()
    perfil.loc[0, "balance"] = np.nan
    with pytest.raises(ValueError):
        p3.construir("kmeans", 3).fit(perfil[columnas_perfil(perfil)])


# --- Validación de perfiles (reglas propias de validacion_p3) ----------------------------------

def test_validacion_usa_la_misma_prueba_que_el_modelo():
    assert validacion_p3.INICIO_PRUEBA == p3.INICIO_PRUEBA


def test_revisar_perfiles_validos():
    perfil = perfil_sintetico()
    nuevo = perfil.tail(1).assign(estacion_id=999)
    r = validacion_p3.revisar_perfiles(perfil, pd.concat([perfil.iloc[1:], nuevo]))
    assert r["integridad_ok"]
    assert (r["estaciones_nuevas_en_test"], r["estaciones_sin_actividad_en_test"]) == (1, 1)


@pytest.mark.parametrize(
    "dano,clave",
    [
        (lambda p: pd.concat([p, p.head(1)]), "ids_duplicados_test"),
        (lambda p: p.assign(balance=p["balance"].where(p.index > 0)), "nulos_test"),
        (lambda p: p.assign(prop_fin_semana=p["prop_fin_semana"] + 1), "fuera_de_rango_test"),
        (lambda p: p.assign(log_viajes_dia=-p["log_viajes_dia"]), "fuera_de_rango_test"),
    ],
    ids=["id_duplicado", "nulos", "prop_fin_semana_mayor_a_1", "log_viajes_negativo"],
)
def test_revisar_perfiles_detecta_problemas(dano, clave):
    perfil = perfil_sintetico()
    r = validacion_p3.revisar_perfiles(perfil, dano(perfil))
    assert not r["integridad_ok"]
    assert r[clave] if not isinstance(r[clave], dict) else any(r[clave].values())
