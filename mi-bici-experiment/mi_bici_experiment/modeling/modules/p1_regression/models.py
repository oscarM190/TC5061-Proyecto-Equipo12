import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mi_bici_experiment.config import (
    EQUIPO,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_COL,
    TRIPS_PARQUET,
    FIGURES_DIR,
    LAGS,
    MLFLOW_EXPERIMENT,
    OBJETIVO_COL,
    PM_DIAS,
    RANDOM_STATE,
    STATION_COL,
    TEST_PARTICION,
    VAL_PARTICION,
)
from mi_bici_experiment.features import (
    agregar_variables,
    construir_viajes_diarios,
    limpiar_viajes,
)
from mi_bici_experiment.tracking import git_info, version_datos


class RegressionModel:
    """P1: baseline de regresión lineal para viajes diarios por estación.
    Métricas usadas: MAE y RMSE."""

    def __init__(
        self,
        fecha_inicio=FECHA_INICIO,
        fecha_fin=FECHA_FIN,
        fraccion_val=VAL_PARTICION,
        fraccion_test=TEST_PARTICION,
    ):
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin
        self.fraccion_test = fraccion_test
        self.num_var = [f"trips_lag_{r}" for r in LAGS] + [
            f"trips_roll_mean_{PM_DIAS}",
            "is_weekend",
        ]
        self.cat_var = [STATION_COL, "dayofweek", "month"]
        self.todas_var = self.num_var + self.cat_var
        self.entrenamiento = None
        self.prueba = None
        self.modelo = None
        self.y_pred = None
        self.fraccion_val = fraccion_val
        self.validacion = None
        self.metricas = {}

    def cargar_viajes(self):  # Dataframe
        columnas = [STATION_COL, FECHA_INICIO_COL]
        viajes = pd.read_parquet(TRIPS_PARQUET, columns=columnas)

        inicio = pd.Timestamp(self.fecha_inicio)
        fin = pd.Timestamp(self.fecha_fin) + pd.Timedelta(days=1)
        return viajes[(viajes[FECHA_INICIO_COL] >= inicio) & (viajes[FECHA_INICIO_COL] < fin)]

    def preparar_datos(self):
        """Limpia, agrega por estación y día, y hace la partición cronológica."""
        viajes = limpiar_viajes(self.cargar_viajes(), STATION_COL, FECHA_INICIO_COL)

        diario = construir_viajes_diarios(viajes, STATION_COL, FECHA_INICIO_COL)
        dataset = agregar_variables(diario, STATION_COL)

        fechas = np.sort(dataset["date"].unique())
        corte_val = fechas[int(len(fechas) * (1 - self.fraccion_val - self.fraccion_test))]
        corte_test = fechas[int(len(fechas) * (1 - self.fraccion_test))]
        self.entrenamiento = dataset[dataset["date"] < corte_val]
        self.validacion = dataset[(dataset["date"] >= corte_val) & (dataset["date"] < corte_test)]
        self.prueba = dataset[dataset["date"] >= corte_test]
        print(
            f"Estaciones: {dataset[STATION_COL].nunique()} | "
            f"filas train: {len(self.entrenamiento):,} | filas test: {len(self.prueba):,}"
        )

    def construir_modelo(self):  # Pipeline
        """Modelo baseline: escalado numérico, one-hot encoding y regresión lineal."""
        preprocesador = ColumnTransformer(
            [
                ("num", StandardScaler(), self.num_var),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_var),
            ]
        )
        return Pipeline(
            [
                ("preprocesamiento", preprocesador),
                ("modelo", LinearRegression()),
            ]
        )

    def entrenar(self):
        """Entrena el modelo con el periodo de entrenamiento."""
        self.modelo = self.construir_modelo()
        self.modelo.fit(self.entrenamiento[self.todas_var], self.entrenamiento[OBJETIVO_COL])

    def evaluar(self):
        """MAE y RMSE sobre validación y prueba."""
        self.metricas = {}
        for prefijo, df in (("val", self.validacion), ("test", self.prueba)):
            y_pred = np.clip(
                self.modelo.predict(df[self.todas_var]), 0, None
            )  # no hay viajes negativos
            y_real = df[OBJETIVO_COL]
            self.metricas[f"{prefijo}_mae"] = float(mean_absolute_error(y_real, y_pred))
            self.metricas[f"{prefijo}_rmse"] = float(np.sqrt(mean_squared_error(y_real, y_pred)))
            if prefijo == "test":
                self.y_pred = y_pred
        print(f"P1 baseline regresión lineal: {self.metricas}")
        return self.metricas

    def graficar(self):  # string
        """Generar gráficas (artefacto): viajes diarios totales reales vs predichos en el periodo de prueba."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        por_dia = (
            self.prueba[["date"]]
            .assign(real=self.prueba[OBJETIVO_COL].to_numpy(), predicho=self.y_pred)
            .groupby("date")
            .sum()
        )
        ax = por_dia.plot(title="P1 regresión lineal: viajes diarios totales (test)")
        ruta = FIGURES_DIR / "Prediccion_regresion_contra_reales.png"
        ax.figure.savefig(ruta)
        plt.close(ax.figure)
        return str(ruta)

    def registrar_mlflow(self):
        """Registra parámetros, métricas y el artefacto en MLflow."""
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(
            run_name="P1-regresion-lineal-baseline",
            tags={
                "equipo": EQUIPO,
                "pregunta": "P1",
                "tipo_modelo": "baseline",
                **git_info(),
                "entorno": os.getenv("MIBICI_ENTORNO", "local"),
            },
        ):
            mlflow.log_params(
                {
                    "fecha_inicio": self.fecha_inicio,
                    "fecha_fin": self.fecha_fin,
                    "fraccion_validacion": self.fraccion_val,
                    "fraccion_test": self.fraccion_test,
                    "data_version": version_datos(),
                    "random_state": RANDOM_STATE,
                    "modelo": "LinearRegression",
                    "lags": str(LAGS),
                    "PM_DIAS": PM_DIAS,
                }
            )

            mlflow.log_metrics(self.metricas)
            mlflow.log_artifact(self.graficar())

    def ejecutar(self):
        self.preparar_datos()
        self.entrenar()
        self.evaluar()
        self.registrar_mlflow()
