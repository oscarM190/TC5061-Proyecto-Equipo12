import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mi_bici_experiment.config import (
    EQUIPO,
    MLFLOW_EXPERIMENT,
    OBJETIVO_COL,
    P1_NOMBRE_REGISTRO,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    STATION_COL)
from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionModel
from mi_bici_experiment.tracking import git_info, version_datos

ACTUAL_PARQUET = PROCESSED_DATA_DIR/"p1_2026_10.parquet"
MIN_MEJORA_PCT = 1.0
N_BOOTSTRAP = 2000


def cargar_actual(ruta: Path = ACTUAL_PARQUET) -> pd.DataFrame:
    """Filas del periodo actual con las variables del modelo y el objetivo, sin nulos."""
    columnas = RegressionModel().todas_var + [OBJETIVO_COL]
    actual = pd.read_parquet(ruta)
    faltantes = [c for c in columnas if c not in actual.columns]
    if faltantes:
        raise ValueError(f"Al conjunto actual le faltan columnas: {faltantes}")
    actual = actual[columnas].dropna().reset_index(drop=True)
    if actual.empty:
        raise ValueError(f"El conjunto actual {ruta} no tiene filas")
    return actual


def metricas(y_real,y_pred) -> dict:
    """MAE y RMSE, con predicciones recortadas a 0 como en la evaluación de P1."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
    }


def intervalo_bootstrap(diferencia, estaciones,n_boot=N_BOOTSTRAP, nivel=0.95,semilla=RANDOM_STATE) -> tuple[float, float]:
    """Intervalo de la diferencia media remuestreando estaciones completas."""
    por_estacion = (
        pd.DataFrame({"d": np.asarray(diferencia), "e": np.asarray(estaciones)})
        .groupby("e")["d"].agg(["sum", "count"]))
    sumas = por_estacion["sum"].to_numpy()
    cuentas = por_estacion["count"].to_numpy()
    rng = np.random.default_rng(semilla)
    indices = rng.integers(0, len(sumas), size=(n_boot, len(sumas)))
    medias = sumas[indices].sum(axis=1) / cuentas[indices].sum(axis=1)
    alfa = (1-nivel) / 2
    return float(np.quantile(medias, alfa)), float(np.quantile(medias,1- alfa))


def comparar(y_real, pred_champion,pred_challenger, estaciones,min_mejora_pct=MIN_MEJORA_PCT) -> dict:
    """En esta parte gana el challenger si (1) el límite inferior del IC 95 % de la reducción del error absoluto es > 0,
    (2) el MAE mejora al menos `min_mejora_pct` y (3) el RMSE no empeora.
    En cualquier otro caso se conserva el champion que se tiene."""
    y_real = np.asarray(y_real, dtype=float)
    pc = np.clip(np.asarray(pred_champion, dtype=float), 0, None)
    pn = np.clip(np.asarray(pred_challenger, dtype=float), 0, None)
    m_champion, m_challenger = metricas(y_real, pc), metricas(y_real, pn)

    # > 0 significa que el challenger se equivoca menos en esa fila.
    diferencia = np.abs(y_real - pc) - np.abs(y_real - pn)
    ic_bajo, ic_alto = intervalo_bootstrap(diferencia, estaciones)
    mejora_mae = (m_champion["mae"] - m_challenger["mae"]) / m_champion["mae"] * 100
    mejora_rmse = (m_champion["rmse"] - m_challenger["rmse"]) / m_champion["rmse"] * 100

    significativa = ic_bajo > 0
    suficiente = mejora_mae >= min_mejora_pct
    rmse_ok = mejora_rmse >= 0
    gana_challenger = significativa and suficiente and rmse_ok
    if gana_challenger:
        razon = (f"El challenger reduce el MAE {mejora_mae:.2f} % (IC 95% de la diferencia: "
                 f"[{ic_bajo:.4f}, {ic_alto:.4f}] viajes, todo > 0) y el RMSE no empeora.")
    else:
        fallos = []
        if not significativa:
            fallos.append(f"la mejora no es significativa (IC 95 %: [{ic_bajo:.4f}, {ic_alto:.4f}])")
        if not suficiente:
            fallos.append(f"el MAE mejora {mejora_mae:.2f} % (< {min_mejora_pct} % requerido)")
        if not rmse_ok:
            fallos.append(f"el RMSE empeora {-mejora_rmse:.2f} %")
        razon = "Se conserva el champion: " + "; ".join(fallos) + "."

    return {
        "n_filas": int(len(y_real)),
        "n_estaciones": int(pd.Series(np.asarray(estaciones)).nunique()),
        "champion": m_champion,
        "challenger": m_challenger,
        "mejora_mae_pct": float(mejora_mae),
        "mejora_rmse_pct": float(mejora_rmse),
        "diferencia_media_error_abs": float(diferencia.mean()),
        "ic95_diferencia": [ic_bajo, ic_alto],
        "pct_filas_gana_challenger": float((diferencia > 0).mean() * 100),
        "pct_filas_gana_champion": float((diferencia < 0).mean() * 100),
        "pct_empates": float((diferencia == 0).mean() * 100),
        "min_mejora_pct": float(min_mejora_pct),
        "decision": "challenger" if gana_challenger else "champion",
        "razon": razon}


def _mae_por_grupo(actual,pred_champion,pred_challenger, columna) -> dict:
    """MAE de cada versión por valor de `columna` (p. ej. día de la semana)."""
    df = pd.DataFrame({
        columna: actual[columna].to_numpy(),
        "champion": np.abs(actual[OBJETIVO_COL].to_numpy() - np.clip(pred_champion, 0, None)),
        "challenger": np.abs(actual[OBJETIVO_COL].to_numpy() - np.clip(pred_challenger, 0, None)),
    })
    return {str(k): {"champion": float(v["champion"]), "challenger": float(v["challenger"])}
            for k, v in df.groupby(columna)[["champion", "challenger"]].mean().iterrows()}


def _grafica(resultado: dict, diferencia_por_dia: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    nombres = ["mae", "rmse"]
    x = np.arange(len(nombres))
    ax1.bar(x - 0.2, [resultado["champion"][n] for n in nombres], 0.4, label="champion")
    ax1.bar(x + 0.2, [resultado["challenger"][n] for n in nombres], 0.4, label="challenger")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres)
    ax1.set_ylabel("Error (viajes por estación y día)")
    ax1.set_title("Periodo actual: menor es mejor")
    ax1.legend()
    dias = sorted(diferencia_por_dia, key=int)
    ax2.bar(dias, [diferencia_por_dia[d]["champion"]-diferencia_por_dia[d]["challenger"] for d in dias])
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Día de la semana (0 = lunes)")
    ax2.set_ylabel("MAE champion - MAE challenger")
    ax2.set_title("> 0: gana el challenger")
    fig.tight_layout()
    return fig


def promover_challenger(client: MlflowClient, version_champion: str, version_challenger: str):
    """Challenger pasa a champion; el champion anterior queda con el alias `previous`."""
    client.set_registered_model_alias(P1_NOMBRE_REGISTRO, "previous", version_champion)
    client.set_registered_model_alias(P1_NOMBRE_REGISTRO, "champion", version_challenger)
    client.delete_registered_model_alias(P1_NOMBRE_REGISTRO, "challenger")


def evaluar_versiones(actual_path: Path = ACTUAL_PARQUET,min_mejora_pct=MIN_MEJORA_PCT,promover=False) -> dict:
    """A/B test de las versiones `champion` y `challenger` de P1 sobre el periodo actual."""
    client = MlflowClient()
    versiones = {}
    for alias in ("champion", "challenger"):
        try:
            versiones[alias] = client.get_model_version_by_alias(P1_NOMBRE_REGISTRO, alias)
        except mlflow.exceptions.MlflowException as error:
            raise ValueError(f"{P1_NOMBRE_REGISTRO} no tiene versión {alias}; "
                             "ejecuta antes la búsqueda y el reentrenamiento de P1") from error

    actual = cargar_actual(actual_path)
    X = actual[RegressionModel().todas_var]
    predicciones = {
        alias: mlflow.sklearn.load_model(f"models:/{P1_NOMBRE_REGISTRO}/{v.version}").predict(X)
        for alias, v in versiones.items()
    }
    resultado = comparar(actual[OBJETIVO_COL], predicciones["champion"],
                         predicciones["challenger"], actual[STATION_COL], min_mejora_pct)
    resultado["versiones"] = {a: str(v.version) for a, v in versiones.items()}
    resultado["meses_actuales"] = sorted(int(m) for m in actual["month"].unique())
    por_dia = _mae_por_grupo(actual, predicciones["champion"], predicciones["challenger"],
                             "dayofweek")
    resultado["mae_por_dia_semana"] = por_dia
    resultado["promovido"] = False

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="P1-ab-test") as run:
        mlflow.set_tags({
            "equipo": EQUIPO, "pregunta": "P1", "stage_of_work": "ab_test",
            "version_champion": resultado["versiones"]["champion"],
            "version_challenger": resultado["versiones"]["challenger"],
            "decision": resultado["decision"],
            **git_info(),
            "entorno": os.getenv("MIBICI_ENTORNO", "local"),
        })
        mlflow.log_params({
            "conjunto_actual": str(actual_path.name if isinstance(actual_path, Path)
                                   else actual_path),
            "data_version": version_datos(),
            "min_mejora_pct": min_mejora_pct,
            "n_filas": resultado["n_filas"],
        })
        mlflow.log_metrics({
            **{f"champion_{k}": v for k, v in resultado["champion"].items()},
            **{f"challenger_{k}": v for k, v in resultado["challenger"].items()},
            "mejora_mae_pct": resultado["mejora_mae_pct"],
            "mejora_rmse_pct": resultado["mejora_rmse_pct"],
            "ic95_bajo": resultado["ic95_diferencia"][0],
            "ic95_alto": resultado["ic95_diferencia"][1],
            "pct_filas_gana_challenger": resultado["pct_filas_gana_challenger"],
        })
        figura = _grafica(resultado, por_dia)
        mlflow.log_figure(figura, "plots/ab_test.png")
        plt.close(figura)

        if promover and resultado["decision"] == "challenger":
            promover_challenger(client, resultado["versiones"]["champion"],
                                resultado["versiones"]["challenger"])
            resultado["promovido"] = True
        mlflow.log_dict(resultado, "tables/ab_test.json")

    for alias, v in versiones.items():
        client.set_model_version_tag(P1_NOMBRE_REGISTRO, v.version, "ab_test_run", run.info.run_id)
    print(f"A/B P1: champion v{resultado['versiones']['champion']} vs "
          f"challenger v{resultado['versiones']['challenger']} -> {resultado['decision']}")
    print(resultado["razon"])
    if resultado["promovido"]:
        print("challenger promovido a champion (el anterior quedó como 'previous')")
    elif resultado["decision"] == "challenger":
        print("El challenger gana; para promoverlo repite con --promover true")
    return resultado


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A/B test de P1: champion vs challenger")
    parser.add_argument("--actual", type=Path, default=ACTUAL_PARQUET,
                        help="Parquet del periodo actual (default: p1_2026_10.parquet)")
    parser.add_argument("--min_mejora_pct", type=float, default=MIN_MEJORA_PCT)
    parser.add_argument("--promover", choices=["true", "false"], default="false")
    args = parser.parse_args(argv)
    evaluar_versiones(args.actual, args.min_mejora_pct, args.promover == "true")


if __name__ == "__main__":
    main()
