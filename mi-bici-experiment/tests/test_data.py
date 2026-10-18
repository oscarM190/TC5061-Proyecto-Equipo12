import pandas as pd

from mi_bici_experiment.config import (
    EQUIPO,
    FECHA_FIN,
    FECHA_INICIO,
    MI_BICI_CSVS,
    MLFLOW_EXPERIMENT,
    P3_FIN_ENTRENAMIENTO,
)


def test_experimento_incluye_equipo():
    assert MLFLOW_EXPERIMENT == f"mibici-equipo{EQUIPO}"


def test_rango_de_fechas_coherente():
    inicio, fin = pd.Timestamp(FECHA_INICIO), pd.Timestamp(FECHA_FIN)
    assert inicio < pd.Timestamp(P3_FIN_ENTRENAMIENTO) < fin


def test_csvs_cubren_el_rango():
    meses = {u.rsplit("_", 2)[-2] + "-" + u.rsplit("_", 1)[-1][:2] for u in MI_BICI_CSVS if "datos_abiertos" in u}
    esperados = {p.strftime("%Y-%m") for p in pd.period_range(FECHA_INICIO, FECHA_FIN, freq="M")}
    assert meses == esperados
