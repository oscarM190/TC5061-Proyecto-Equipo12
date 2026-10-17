"""Pruebas de las variables de P3 (clustering de estaciones) con datos sintéticos."""

import numpy as np
import pandas as pd

from mi_bici_experiment.config import (
    DESTINO_COL,
    FECHA_FIN_COL,
    FECHA_INICIO_COL,
    STATION_COL,
)
from mi_bici_experiment.features import (
    filtrar_viajes_validos,
    perfil_estaciones,
    perfil_horario,
    resumen_uso,
)


def viajes_sinteticos(n=3000, seed=0):
    """Viajes entre 4 estaciones durante dos meses, con duraciones de 5 a 40 minutos."""
    rng = np.random.default_rng(seed)
    inicio = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 59 * 24 * 60, n), "min")
    return pd.DataFrame(
        {
            STATION_COL: rng.integers(1, 5, n),
            DESTINO_COL: rng.integers(1, 5, n),
            FECHA_INICIO_COL: inicio,
            FECHA_FIN_COL: inicio + pd.to_timedelta(rng.integers(5, 40, n), "min"),
        }
    )


def test_perfil_horario_suma_uno():
    perfil = perfil_horario(viajes_sinteticos(), STATION_COL, FECHA_INICIO_COL, "sal")
    assert list(perfil.columns) == [f"sal_h{h:02d}" for h in range(24)]
    assert np.allclose(perfil.sum(axis=1), 1)


def test_filtra_duraciones_y_devoluciones():
    t = pd.Timestamp("2026-01-05 08:00")
    viajes = pd.DataFrame(
        {
            STATION_COL: [1, 1, 1, 1],
            DESTINO_COL: [2, 2, 2, 1],
            FECHA_INICIO_COL: [t] * 4,
            FECHA_FIN_COL: [
                t + pd.Timedelta(minutes=15),  # válido
                t - pd.Timedelta(minutes=1),  # duración negativa
                t + pd.Timedelta(hours=7),  # más de 6 h
                t + pd.Timedelta(seconds=30),  # devolución inmediata
            ],
        }
    )
    assert len(filtrar_viajes_validos(viajes)) == 1


def test_resumen_en_rango():
    resumen = resumen_uso(viajes_sinteticos())
    assert resumen["balance"].between(-1, 1).all()
    assert resumen["prop_fin_semana"].between(0, 1).all()


def test_perfil_no_usa_datos_posteriores():
    """Cambiar los viajes posteriores al fin del entrenamiento no altera el perfil."""
    viajes = viajes_sinteticos()
    fin = "2026-01-31"
    base = perfil_estaciones(viajes, "2026-01-01", fin, min_viajes=1)

    alterados = viajes.copy()
    futuros = alterados[FECHA_INICIO_COL] > pd.Timestamp(fin) + pd.Timedelta(days=1)
    alterados.loc[futuros, STATION_COL] = 1
    alterados.loc[futuros, FECHA_INICIO_COL] += pd.Timedelta(hours=3)
    pd.testing.assert_frame_equal(
        base, perfil_estaciones(alterados, "2026-01-01", fin, min_viajes=1)
    )
