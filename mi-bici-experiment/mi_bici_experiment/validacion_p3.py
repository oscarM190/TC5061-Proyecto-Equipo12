"""P3: validación con Deepchecks de los perfiles por estación que recibe el modelo de clustering.

`data_validation.py` revisa los viajes; aquí se revisa la tabla que realmente entra a P3 (una
fila por estación con prop_fin_semana, log_viajes_dia y balance) en entrenamiento y en prueba.
Deepchecks 0.19.1 solo carga en Python 3.10 y `features.py` necesita el entorno del proyecto,
por eso son dos pasos:

    python -m mi_bici_experiment.validacion_p3 exportar   # entorno del proyecto
    python -m mi_bici_experiment.validacion_p3 validar    # entorno de Deepchecks

`run_deepchecks.sh` / `.ps1` ejecutan `validar` después de la validación de viajes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mi_bici_experiment.config import (
    FECHA_FIN,
    P3_FIN_ENTRENAMIENTO,
    P3_INICIO_ENTRENAMIENTO,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

PERFIL_TRAIN = PROCESSED_DATA_DIR / "p3_perfil_train.parquet"
PERFIL_TEST = PROCESSED_DATA_DIR / "p3_perfil_test.parquet"
REPORTE_DIR = REPORTS_DIR / "deepchecks" / "p3"
# Mismo inicio de prueba que models.INICIO_PRUEBA (models importa mlflow, que no está en el
# entorno de Deepchecks); test_p3.py comprueba que coincidan.
INICIO_PRUEBA = str((pd.Timestamp(P3_FIN_ENTRENAMIENTO) + pd.Timedelta(days=1)).date())
RANGOS = {"prop_fin_semana": (0.0, 1.0), "balance": (-1.0, 1.0), "log_viajes_dia": (0.0, None)}
MAX_DRIFT = 0.2


def exportar(train_path: Path = PERFIL_TRAIN, test_path: Path = PERFIL_TEST) -> None:
    """Construye los perfiles de entrenamiento y prueba con las mismas funciones que el modelo."""
    from mi_bici_experiment.config import (
        DESTINO_COL,
        FECHA_FIN_COL,
        FECHA_INICIO_COL,
        STATION_COL,
        TRIPS_PARQUET,
    )
    from mi_bici_experiment.features import filtrar_viajes_validos, perfil_estaciones

    cols = [STATION_COL, DESTINO_COL, FECHA_INICIO_COL, FECHA_FIN_COL]
    viajes = filtrar_viajes_validos(pd.read_parquet(TRIPS_PARQUET, columns=cols))
    train_path.parent.mkdir(parents=True, exist_ok=True)
    perfil_estaciones(viajes, P3_INICIO_ENTRENAMIENTO, P3_FIN_ENTRENAMIENTO).to_parquet(train_path)
    perfil_estaciones(viajes, INICIO_PRUEBA, FECHA_FIN).to_parquet(test_path)


def revisar_perfiles(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Reglas propias de P3 que Deepchecks no cubre: ids únicos, rangos válidos y estaciones
    que aparecen o desaparecen entre periodos (las nuevas no tienen grupo de entrenamiento)."""
    resultado = {"estaciones_train": len(train), "estaciones_test": len(test)}
    for nombre, perfil in (("train", train), ("test", test)):
        resultado[f"ids_duplicados_{nombre}"] = int(perfil["estacion_id"].duplicated().sum())
        resultado[f"nulos_{nombre}"] = int(perfil.isna().sum().sum())
        fuera = {}
        for col, (bajo, alto) in RANGOS.items():
            valores = perfil[col]
            n = (valores < bajo).sum() + (0 if alto is None else (valores > alto).sum())
            fuera[col] = int(n)
        resultado[f"fuera_de_rango_{nombre}"] = fuera
    ids_train, ids_test = set(train["estacion_id"]), set(test["estacion_id"])
    resultado["estaciones_nuevas_en_test"] = len(ids_test - ids_train)
    resultado["estaciones_sin_actividad_en_test"] = len(ids_train - ids_test)
    resultado["integridad_ok"] = not any(
        resultado[f"ids_duplicados_{n}"] or resultado[f"nulos_{n}"]
        or any(resultado[f"fuera_de_rango_{n}"].values())
        for n in ("train", "test")
    )
    return resultado


def validar(train_path: Path = PERFIL_TRAIN, test_path: Path = PERFIL_TEST,
            reporte_dir: Path = REPORTE_DIR) -> dict:
    """Integridad de cada periodo y drift entrenamiento → prueba de las variables del modelo.

    Las condiciones de integridad (sin nulos, sin perfiles duplicados, sin valor único) deciden
    el código de salida. El drift se reporta: un cambio estacional entre los 12 meses de
    entrenamiento y el trimestre de prueba es esperable y es lo que mide el ARI de estabilidad.
    """
    try:
        from deepchecks.tabular import Dataset, Suite
        from deepchecks.tabular.checks import (
            DataDuplicates,
            FeatureDrift,
            IsSingleValue,
            MixedDataTypes,
            MultivariateDrift,
            OutlierSampleDetection,
            PercentOfNulls,
        )
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            "Deepchecks no cargó. Usa Python 3.10 con requirements-deepchecks.txt "
            "(run_deepchecks.sh / run_deepchecks.ps1)."
        ) from error

    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    resultado = revisar_perfiles(train, test)

    def dataset(perfil, nombre):
        return Dataset(perfil.set_index("estacion_id"), cat_features=[], dataset_name=nombre)

    integridad = Suite(
        "P3 integridad de perfiles",
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0),
        MixedDataTypes(),
        IsSingleValue().add_condition_not_single_value(),
        DataDuplicates().add_condition_ratio_less_or_equal(0),
        OutlierSampleDetection(),
    )
    drift = Suite(
        "P3 drift entrenamiento vs prueba",
        FeatureDrift().add_condition_drift_score_less_than(max_allowed_numeric_score=MAX_DRIFT),
        MultivariateDrift().add_condition_overall_drift_value_less_than(MAX_DRIFT),
    )
    # Drift con todas las estaciones de prueba y solo con las que ya tenían grupo: separa el efecto
    # de las estaciones nuevas del cambio de uso en las existentes.
    comunes = test[test["estacion_id"].isin(train["estacion_id"])]
    reporte_dir.mkdir(parents=True, exist_ok=True)
    corridas = {
        "integridad_train": integridad.run(dataset(train, "Train")),
        "integridad_test": integridad.run(dataset(test, "Test")),
        "drift": drift.run(train_dataset=dataset(train, "Train"),
                           test_dataset=dataset(test, "Test")),
        "drift_estaciones_comunes": drift.run(train_dataset=dataset(train, "Train"),
                                              test_dataset=dataset(comunes, "Test comunes")),
    }
    for nombre, corrida in corridas.items():
        html = reporte_dir / f"{nombre}.html"
        html.unlink(missing_ok=True)  # Deepchecks crea "nombre (1).html" si el archivo ya existe
        corrida.save_as_html(str(html), as_widget=False)
        resultado[f"deepchecks_{nombre}_ok"] = corrida.passed(fail_if_check_not_run=True)
        if nombre.startswith("drift"):
            valores = {r.get_header(): r.value for r in corrida.results}
            resultado[f"{nombre}_por_variable"] = {
                col: round(float(v["Drift score"]), 3)
                for col, v in valores["Feature Drift"].items()
            }
            resultado[f"{nombre}_multivariado"] = round(
                float(valores["Multivariate Drift"]["domain_classifier_drift_score"]), 3
            )
    (reporte_dir / "audit_p3.json").write_text(json.dumps(resultado, indent=2) + "\n",
                                               encoding="utf-8")
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paso", choices=["exportar", "validar"])
    args = parser.parse_args()
    if args.paso == "exportar":
        exportar()
        print(f"Perfiles P3: {PERFIL_TRAIN}, {PERFIL_TEST}")
        return
    if not PERFIL_TRAIN.is_file():
        parser.error(f"No existe {PERFIL_TRAIN}; ejecuta antes el paso `exportar`")
    resultado = validar()
    print(json.dumps(resultado, indent=2))
    print(f"Reportes Deepchecks P3: {REPORTE_DIR}")
    integridad_ok = resultado["integridad_ok"] and all(
        resultado[f"deepchecks_integridad_{n}_ok"] for n in ("train", "test")
    )
    if not integridad_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
