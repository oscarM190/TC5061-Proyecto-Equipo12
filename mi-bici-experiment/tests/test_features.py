import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import mi_bici_experiment.features as features
from mi_bici_experiment.config import DESTINO_COL, FECHA_FIN_COL, FECHA_INICIO_COL, STATION_COL


# [TEST 1] Dates are parsed, invalid rows are removed, and the input is unchanged.
def test_limpiar_viajes_parses_dates_and_preserves_input():
    viajes = pd.DataFrame({"station": [1, 2, None, 3], "when": ["2024-10-27", "invalid", "2024-10-28", None]})
    original = viajes.copy(deep=True)

    actual = features.limpiar_viajes(viajes, "station", "when")

    expected = pd.DataFrame({"station": [1.0], "when": pd.to_datetime(["2024-10-27"])})
    assert_frame_equal(actual, expected)
    assert_frame_equal(viajes, original)


# [TEST 2] Daily trips combine hours, fill gaps, and begin at each station's first trip.
def test_construir_viajes_diarios_counts_and_fills():
    viajes = pd.DataFrame({
        "station": [1, 1, 1, 2, 2],
        "when": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 20:00", "2024-01-03 10:00", "2024-01-02 09:00", "2024-01-04 09:00"]),
    })
    expected = pd.DataFrame({
        "station": [1, 1, 1, 1, 2, 2, 2],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "trips": [2, 0, 1, 0, 1, 0, 1],
    })

    assert_frame_equal(features.construir_viajes_diarios(viajes, "station", "when"), expected)


# [TEST 3] P1 lags and seven-day means use only previous days of the same station.
def test_agregar_variables_uses_previous_days_per_station():
    dates = pd.date_range("2024-01-01", periods=9)
    diario = pd.DataFrame({
        "station": [1] * 9 + [2] * 9,
        "date": list(dates) * 2,
        "trips": list(range(1, 10)) + list(range(101, 110)),
    }).sample(frac=1, random_state=42).reset_index(drop=True)
    expected = pd.DataFrame({
        "station": [1, 1, 2, 2],
        "date": pd.to_datetime(["2024-01-08", "2024-01-09"] * 2),
        "trips": [8, 9, 108, 109],
        "trips_lag_1": [7.0, 8.0, 107.0, 108.0],
        "trips_lag_7": [1.0, 2.0, 101.0, 102.0],
        "trips_roll_mean_7": [4.0, 5.0, 104.0, 105.0],
        "dayofweek": [0, 1, 0, 1],
        "month": [1, 1, 1, 1],
        "is_weekend": [0, 0, 0, 0],
    })

    assert_frame_equal(features.agregar_variables(diario, "station"), expected, check_dtype=False)


# [TEST 4] Daily balances use arrival dates and fill only each station's active period.
def test_agregar_balance_counts_overnight_arrivals_and_active_days(monkeypatch):
    # Patch the name used by the feature module so the test isolates calculations.
    monkeypatch.setattr(features, "print_balance_normalizado", lambda frame: None)
    viajes = pd.DataFrame({
        "Inicio_del_viaje": pd.to_datetime(["2024-01-01 23:00", "2024-01-03 10:00", "2024-01-04 10:00"]),
        "Fin_del_viaje": pd.to_datetime(["2024-01-02 01:00", "2024-01-03 11:00", "2024-01-04 11:00"]),
        "Origen_Id": [1, 1, 2], "Destino_Id": [2, 2, 1], "dpcapacity": [2, 2, 4],
    })
    expected = pd.DataFrame({
        "Fecha": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "Salidas": [1., 0., 1., 0., 0., 0., 1.],
        "Llegadas": [0., 0., 0., 1., 1., 1., 0.],
        "Estacion_Id": [1, 1, 1, 1, 2, 2, 2],
        "dpcapacity": [2, 2, 2, 2, 4, 4, 4],
        "Balance_Neto": [-1., 0., -1., 1., 1., 1., -1.],
        "Balance_Normalizado": [-0.5, 0., -0.5, 0.5, 0.25, 0.25, -0.25],
    })

    actual = features.agregar_balance_neto_y_normalizado(viajes).reset_index(drop=True)
    assert_frame_equal(actual, expected, check_dtype=False)


# [TEST 5] Nonpositive, missing, and infinite station capacities are excluded.
@pytest.mark.parametrize("capacity", [0, -1, np.nan, np.inf, -np.inf])
def test_agregar_balance_excludes_invalid_capacities(monkeypatch, capacity):
    monkeypatch.setattr(features, "print_balance_normalizado", lambda frame: None)
    viajes = pd.DataFrame({
        "Inicio_del_viaje": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "Fin_del_viaje": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "Origen_Id": [1, 2], "Destino_Id": [1, 2], "dpcapacity": [10.0, capacity],
    })
    expected = pd.DataFrame({
        "Fecha": pd.to_datetime(["2024-01-01"]), "Salidas": [1], "Llegadas": [1],
        "Estacion_Id": [1], "dpcapacity": [10.0], "Balance_Neto": [0], "Balance_Normalizado": [0.0],
    })

    actual = features.agregar_balance_neto_y_normalizado(viajes).reset_index(drop=True)
    assert_frame_equal(actual, expected, check_dtype=False)


# [TEST 6] Thresholds and extreme balances classify correctly and targets use the exact next day.
def test_clasificacion_thresholds_and_exact_next_day():
    balance = pd.DataFrame({
        "Estacion_Id": [2, 1, 1, 1, 1, 1, 1, 1, 2],
        "Fecha": pd.to_datetime(["2024-02-02", "2024-01-27", "2024-01-28", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-03", "2024-02-01"]),
        "Balance_Normalizado": [0.8, -2.0, -0.45, -0.449, 0.0, 0.449, 0.45, 2.0, -0.8],
    })
    expected = pd.DataFrame({
        "Estacion_Id": [1] * 7 + [2, 2],
        "Fecha": pd.to_datetime(["2024-01-27", "2024-01-28", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-03", "2024-02-01", "2024-02-02"]),
        "Balance_Normalizado": [-2.0, -0.45, -0.449, 0.0, 0.449, 0.45, 2.0, -0.8, 0.8],
        "Estado_Actual": [0, 0, 1, 1, 1, 2, 2, 0, 2],
        "Dia_Semana": [5, 6, 0, 1, 2, 3, 5, 3, 4],
        "Es_Fin_Semana": [1, 1, 0, 0, 0, 0, 1, 0, 0],
        "Mes": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        "Fecha_Objetivo": pd.to_datetime(["2024-01-28", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-04", "2024-02-02", "2024-02-03"]),
        "Dia_Semana_Objetivo": [6, 0, 1, 2, 3, 4, 6, 4, 5],
        "Es_Fin_Semana_Objetivo": [1, 0, 0, 0, 0, 0, 1, 0, 1],
        "Mes_Objetivo": [1, 1, 1, 1, 2, 2, 2, 2, 2],
        "Riesgo_T1": pd.array([0, 1, 1, 1, 2, None, None, 2, None], dtype="Int64"),
    })

    assert_frame_equal(features.clasificacion_variable_objetivo_y_caracterizticas(balance), expected, check_dtype=False)


# Build consecutive daily P2 data with distinct values for each station.
def datos_memoria():
    return pd.DataFrame({
        "Estacion_Id": [1] * 10 + [2] * 10,
        "Fecha": list(pd.date_range("2024-01-01", periods=10)) * 2,
        "dpcapacity": [10.0] * 20,
        "Llegadas": list(range(1, 11)) + list(range(101, 111)),
        "Salidas": list(range(2, 21, 2)) + list(range(202, 221, 2)),
        "Balance_Normalizado": [i / 10 for i in list(range(1, 11)) + list(range(101, 111))],
        "Riesgo_T1": pd.array([1] * 8 + [None, 2] + [1] * 8 + [None, 2], dtype="Int64"),
    })


# [TEST 7] P2 lags are station-specific and rolling windows include today, even after a missing target.
def test_memoria_historica_exact_lags_and_rolling_values():
    datos = datos_memoria().sample(frac=1, random_state=42).reset_index(drop=True)
    expected = pd.DataFrame({
        "Estacion_Id": [1, 1, 2, 2],
        "Fecha": pd.to_datetime(["2024-01-08", "2024-01-10"] * 2),
        "dpcapacity": [10.0] * 4,
        "Llegadas": [8, 10, 108, 110], "Salidas": [16, 20, 216, 220],
        "Balance_Normalizado": [0.8, 1.0, 10.8, 11.0], "Riesgo_T1": [1, 2, 1, 2],
        "Llegadas_Por_Capacidad": [0.8, 1.0, 10.8, 11.0],
        "Salidas_Por_Capacidad": [1.6, 2.0, 21.6, 22.0],
        "Balance_Lag_1": [0.7, 0.9, 10.7, 10.9],
        "Balance_Lag_2": [0.6, 0.8, 10.6, 10.8],
        "Balance_Lag_3": [0.5, 0.7, 10.5, 10.7],
        "Balance_Lag_7": [0.1, 0.3, 10.1, 10.3],
        "Media_Movil_3d_Balance": [0.7, 0.9, 10.7, 10.9],
        "Media_Movil_7d_Balance": [0.5, 0.7, 10.5, 10.7],
        "Llegadas_Lag_1": [7., 9., 107., 109.], "Llegadas_Por_Capacidad_Lag_1": [0.7, 0.9, 10.7, 10.9],
        "Salidas_Lag_1": [14., 18., 214., 218.], "Salidas_Por_Capacidad_Lag_1": [1.4, 1.8, 21.4, 21.8],
        "Llegadas_Lag_2": [6., 8., 106., 108.], "Llegadas_Por_Capacidad_Lag_2": [0.6, 0.8, 10.6, 10.8],
        "Salidas_Lag_2": [12., 16., 212., 216.], "Salidas_Por_Capacidad_Lag_2": [1.2, 1.6, 21.2, 21.6],
        "Llegadas_Lag_3": [5., 7., 105., 107.], "Llegadas_Por_Capacidad_Lag_3": [0.5, 0.7, 10.5, 10.7],
        "Salidas_Lag_3": [10., 14., 210., 214.], "Salidas_Por_Capacidad_Lag_3": [1.0, 1.4, 21.0, 21.4],
        "Llegadas_Lag_7": [1., 3., 101., 103.], "Llegadas_Por_Capacidad_Lag_7": [0.1, 0.3, 10.1, 10.3],
        "Salidas_Lag_7": [2., 6., 202., 206.], "Salidas_Por_Capacidad_Lag_7": [0.2, 0.6, 20.2, 20.6],
        "Media_Movil_7d_Llegadas": [5., 7., 105., 107.],
        "Media_Movil_7d_Salidas": [10., 14., 210., 214.],
        "Media_Movil_7d_Llegadas_Por_Capacidad": [0.5, 0.7, 10.5, 10.7],
        "Media_Movil_7d_Salidas_Por_Capacidad": [1.0, 1.4, 21.0, 21.4],
    })

    actual = features.creacion_variables_memoria_historica(datos).reset_index(drop=True)
    assert_frame_equal(actual, expected, check_dtype=False)


# [TEST 8] Duplicate station/date rows and gaps in a station's calendar are rejected.
@pytest.mark.parametrize("problem", ["duplicate", "gap"])
def test_memoria_historica_rejects_invalid_calendar(problem):
    datos = datos_memoria()
    if problem == "duplicate":
        datos = pd.concat([datos, datos.iloc[[0]]], ignore_index=True)
        message = "fila por estación"
    else:
        datos = datos.drop(index=4)
        message = "calendario diario"
    with pytest.raises(ValueError, match=message):
        features.creacion_variables_memoria_historica(datos)


# [TEST 9] P2 history rejects missing, nonpositive, or infinite capacities.
@pytest.mark.parametrize("capacity", [0, -1, np.nan, np.inf, -np.inf])
def test_memoria_historica_rejects_invalid_capacity(capacity):
    datos = datos_memoria()
    datos.loc[0, "dpcapacity"] = capacity
    with pytest.raises(ValueError, match="capacidad positiva y finita"):
        features.creacion_variables_memoria_historica(datos)


# [TEST 10] Required fields and date ordering are checked while same-day trips remain valid.
def test_filtrar_viajes_validos_removes_missing_and_reversed_dates():
    viajes = pd.DataFrame({
        STATION_COL: [1, 1, None, 1, 1, 1, 1], DESTINO_COL: [2, 2, 2, None, 2, 2, 2],
        FECHA_INICIO_COL: pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01", None, "2024-01-01", "2024-01-02"]),
        FECHA_FIN_COL: pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-01", "2024-01-01", "2024-01-01", None, "2024-01-01"]),
    })
    assert_frame_equal(features.filtrar_viajes_validos(viajes), viajes.iloc[[0, 1]])


# [TEST 11] Both period boundaries include their entire days, but the next day is excluded.
def test_filtrar_periodo_includes_both_boundaries():
    viajes = pd.DataFrame({FECHA_INICIO_COL: pd.to_datetime([
        "2024-01-09 23:59:59", "2024-01-10 00:00:00", "2024-01-11 23:59:59", "2024-01-12 00:00:00",
    ])})
    assert_frame_equal(features.filtrar_periodo(viajes, "2024-01-10", "2024-01-11"), viajes.iloc[[1, 2]])


# [TEST 12] Weekend shares, daily usage, and net balances match hand-calculated values.
def test_resumen_uso_exact_values_including_no_arrivals():
    viajes = pd.DataFrame({
        STATION_COL: [1, 1, 1, 2], DESTINO_COL: [1, 2, 2, 2],
        FECHA_INICIO_COL: pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-07"]),
    })
    # Station 3 has departures but no arrivals; station 1 spans three calendar days.
    viajes.loc[3, STATION_COL] = 3
    expected = pd.DataFrame({
        "prop_fin_semana": [2 / 3, 1.0], "log_viajes_dia": [np.log1p(1), np.log1p(1)],
        "balance": [-0.5, -1.0],
    }, index=pd.Index([1, 3], name="estacion_id"))
    assert_frame_equal(features.resumen_uso(viajes), expected, check_dtype=False)


# [TEST 13] The minimum trip count is inclusive and outside-period trips do not affect profiles.
def test_perfil_estaciones_threshold_and_period():
    viajes = pd.DataFrame({
        STATION_COL: [1, 1, 2, 2, 1], DESTINO_COL: [1, 2, 1, 2, 2],
        FECHA_INICIO_COL: pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-06", "2024-01-08", "2024-01-08"]),
    })
    expected = pd.DataFrame({
        "estacion_id": [1], "prop_fin_semana": [1.0], "log_viajes_dia": [np.log1p(1)],
        "balance": [0.0],
    })
    assert_frame_equal(features.perfil_estaciones(viajes, "2024-01-06", "2024-01-07", min_viajes=2), expected, check_dtype=False)


# [TEST 14] Profile column selection excludes station IDs and applies the requested prefix.
def test_columnas_perfil_excludes_id_and_filters_prefix():
    perfil = pd.DataFrame(columns=["estacion_id", "salidas", "sal_media", "balance"])
    assert features.columnas_perfil(perfil) == ["salidas", "sal_media", "balance"]
    assert features.columnas_perfil(perfil, "sal") == ["salidas", "sal_media"]
    assert features.columnas_perfil(perfil, "missing") == []


# [TEST 15] Station metadata is renamed and municipality codes have an OTRO fallback.
def test_info_estaciones_renames_and_extracts_municipality():
    nomenclatura = pd.DataFrame({
        "id": [1, 2, 3], "name": ["Centro (GDL-001)", "Plaza (ZAP-42)", "Sin código"],
        "latitude": [20.1, 20.2, 20.3], "longitude": [-103.1, -103.2, -103.3],
        "dpcapacity": [10, 20, 30], "unused": [1, 2, 3],
    })
    expected = pd.DataFrame({
        "estacion_id": [1, 2, 3], "nombre": ["Centro (GDL-001)", "Plaza (ZAP-42)", "Sin código"],
        "lat": [20.1, 20.2, 20.3], "lon": [-103.1, -103.2, -103.3],
        "capacidad": [10, 20, 30], "municipio": ["GDL", "ZAP", "OTRO"],
    })
    assert_frame_equal(features.info_estaciones(nomenclatura), expected)


# [TEST 16] Values just below, at, and just above each threshold use the correct class.
@pytest.mark.parametrize(
    ("balance", "estado"),
    [(-0.451, 0), (-0.45, 0), (-0.449, 1), (0.449, 1), (0.45, 2), (0.451, 2)],
)
def test_clasificacion_threshold_boundaries(balance, estado):
    diario = pd.DataFrame({
        "Estacion_Id": [1], "Fecha": pd.to_datetime(["2024-01-01"]),
        "Balance_Normalizado": [balance],
    })
    result = features.clasificacion_variable_objetivo_y_caracterizticas(diario)
    assert result["Estado_Actual"].tolist() == [estado]
    assert pd.isna(result.loc[0, "Riesgo_T1"])
