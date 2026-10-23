import pandas as pd

from mi_bici_experiment.config import (
    EQUIPO,
    FECHA_FIN,
    FECHA_INICIO,
    MI_BICI_CSVS,
    MLFLOW_EXPERIMENT,
    P3_FIN_ENTRENAMIENTO,
    P3_INICIO_ENTRENAMIENTO,
)


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
