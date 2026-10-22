"""Pruebas de P3 (clustering de estaciones) con datos sintéticos."""

import json

import numpy as np
import pandas as pd
import pytest

from mi_bici_experiment.config import (
    DESTINO_COL,
    FECHA_FIN,
    FECHA_FIN_COL,
    FECHA_INICIO_COL,
    P3_FIN_ENTRENAMIENTO,
    P3_INICIO_ENTRENAMIENTO,
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
