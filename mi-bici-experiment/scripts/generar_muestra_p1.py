"""Genera serving/seed/p1_muestra.csv: filas reales de P1 para el modelo de demostración.

Toma las filas estación-día del último mes de prueba de P1 (septiembre de 2026), con las mismas
variables que espera la API y los viajes reales del día, y guarda una muestra fija (semilla del proyecto).
scripts/seed_model.py la usa para registrar un champion cuando el Model Registry está vacío, sin
descargar la ingesta completa. También sirve como ejemplo de peticiones válidas para la API.

Uso (con el paquete instalado con `pip install -e .` y el Parquet de la ingesta):
    cd mi-bici-experiment && python scripts/generar_muestra_p1.py --filas 500
"""

from pathlib import Path

import pandas as pd
import typer

from mi_bici_experiment.api import MODEL_FEATURES
from mi_bici_experiment.config import (
    FECHA_INICIO_COL,
    OBJETIVO_COL,
    RANDOM_STATE,
    STATION_COL,
    TRIPS_PARQUET,
)
from mi_bici_experiment.features import (
    agregar_variables,
    construir_viajes_diarios,
    limpiar_viajes,
)

SALIDA = Path(__file__).resolve().parents[2] / "serving" / "seed" / "p1_muestra.csv"


def main(filas: int = 500, desde: str = "2026-08-01", hasta: str = "2026-09-30"):
    viajes = pd.read_parquet(TRIPS_PARQUET, columns=[STATION_COL, FECHA_INICIO_COL])
    inicio, fin = pd.Timestamp(desde), pd.Timestamp(hasta) + pd.Timedelta(days=1)
    viajes = viajes[(viajes[FECHA_INICIO_COL] >= inicio) & (viajes[FECHA_INICIO_COL] < fin)]

    diario = construir_viajes_diarios(
        limpiar_viajes(viajes, STATION_COL, FECHA_INICIO_COL), STATION_COL, FECHA_INICIO_COL
    )
    dataset = agregar_variables(diario, STATION_COL)
    # Solo septiembre: agosto sirve de historia para los rezagos de 7 días.
    dataset = dataset[dataset["date"] >= pd.Timestamp("2026-09-01")]
    muestra = dataset.sample(n=min(filas, len(dataset)), random_state=RANDOM_STATE)
    # La columna objetivo (viajes reales) no forma parte de la petición a la API; la usa seed_model.py.
    muestra[list(MODEL_FEATURES) + [OBJETIVO_COL]].round(4).to_csv(SALIDA, index=False)
    print(f"{len(muestra)} filas guardadas en {SALIDA}")


if __name__ == "__main__":
    typer.run(main)
