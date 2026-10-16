from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mi_bici_experiment.config import MODELS_DIR, PROCESSED_DATA_DIR

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
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_COL,
    FEATURES_PARQUET,
    FIGURES_DIR,
    LAGS,
    MLFLOW_EXPERIMENT,
    OBJETIVO_COL,
    PM_DIAS,
    RANDOM_STATE,
    STATION_COL,
    TEST_PARTICION)
    
from mi_bici_experiment.features import (
    agregar_variables,
    construir_viajes_diarios,
    limpiar_viajes)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


"""P1 (regresión): baseline de regresión lineal para viajes diarios por estación."""
class RegresionP1:
    """P1: baseline de regresión lineal para viajes diarios por estación.
    Métricas usadas: MAE y RMSE."""

    def __init__(self, fecha_inicio=FECHA_INICIO, fecha_fin=FECHA_FIN, fraccion_test=TEST_PARTICION):
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin
        self.fraccion_test = fraccion_test
        self.num_var = [f"trips_lag_{r}" for r in LAGS] + [f"trips_roll_mean_{PM_DIAS}", "is_weekend"]
        self.cat_var = [STATION_COL, "dayofweek", "month"]
        self.todas_var = self.num_var + self.cat_var
        self.entrenamiento = None
        self.prueba = None
        self.modelo = None
        self.y_pred = None
        self.metricas = {}

    def cargar_viajes(self):    #Dataframe
        columnas = [STATION_COL, FECHA_INICIO_COL]
        viajes = pd.read_parquet(FEATURES_PARQUET, columns=columnas)

        inicio = pd.Timestamp(self.fecha_inicio)
        fin = pd.Timestamp(self.fecha_fin) + pd.Timedelta(days=1)
        return viajes[(viajes[FECHA_INICIO_COL] >= inicio) & (viajes[FECHA_INICIO_COL] < fin)]

    def preparar_datos(self):
        """Limpia, agrega por estación y día, y hace la partición cronológica."""
        viajes = limpiar_viajes(self.cargar_viajes(), STATION_COL, FECHA_INICIO_COL)

        diario = construir_viajes_diarios(viajes, STATION_COL, FECHA_INICIO_COL)
        dataset= agregar_variables(diario, STATION_COL)

        fechas = np.sort(dataset["date"].unique())
        corte = fechas[int(len(fechas) *(1 - self.fraccion_test))]
        self.entrenamiento = dataset[dataset["date"] <corte]
        self.prueba = dataset[dataset["date"] >= corte]
        print(f"Estaciones: {dataset[STATION_COL].nunique()} | "
              f"filas train: {len(self.entrenamiento):,} | filas test: {len(self.prueba):,}")

    def construir_modelo(self):   #Pipeline
        """Modelo baseline: escalado numérico, one-hot encoding y regresión lineal."""
        preprocesador = ColumnTransformer([
            ("num", StandardScaler(), self.num_var),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_var),
        ])
        return Pipeline([
            ("preprocesamiento", preprocesador),
            ("modelo", LinearRegression()),
        ])

    def entrenar(self):
        """Entrena el modelo con el periodo de entrenamiento."""
        self.modelo = self.construir_modelo()
        self.modelo.fit(self.entrenamiento[self.todas_var], self.entrenamiento[OBJETIVO_COL])

    def evaluar(self):    #Dict
        """Calculo de MAE y RMSE sobre el conjunto de prueba."""
        y_real = self.prueba[OBJETIVO_COL]
        self.y_pred =np.clip(self.modelo.predict(self.prueba[self.todas_var]), 0, None)  # no hay viajes negativos
        self.metricas = {
            "test_mae":float(mean_absolute_error(y_real, self.y_pred)),
            "test_rmse":float(np.sqrt(mean_squared_error(y_real, self.y_pred)))}
        print(f"P1 baseline regresión lineal: {self.metricas}")
        return self.metricas

    def graficar(self):   #string
        """Generar gráficas (artefacto): viajes diarios totales reales vs predichos en el periodo de prueba."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        por_dia = (self.prueba[["date"]]
                   .assign(real=self.prueba[OBJETIVO_COL].to_numpy(), predicho=self.y_pred)
                   .groupby("date").sum())
        ax = por_dia.plot(title="P1 regresión lineal: viajes diarios totales (test)")
        ruta = FIGURES_DIR/"Prediccion_regresion_contra_reales.png"
        ax.figure.savefig(ruta)
        plt.close(ax.figure)
        return str(ruta)

    def registrar_mlflow(self):
        """Registra parámetros, métricas y el artefacto en MLflow."""
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="P1-regresion-lineal-baseline"):
            mlflow.log_params({
                "fecha_inicio":self.fecha_inicio,
                "fecha_fin":self.fecha_fin,
                "fraccion_test":self.fraccion_test,
                "random_state":RANDOM_STATE,
                "modelo":"LinearRegression",
                "lags":str(LAGS),
                "PM_DIAS":PM_DIAS,
            })
            mlflow.log_metrics(self.metricas)
            mlflow.log_artifact(self.graficar())

    def ejecutar(self):
        self.preparar_datos()
        self.entrenar()
        self.evaluar()
        self.registrar_mlflow()


@app.command()
def regresion_p1():
    """Ejecuta el baseline de regresión (P1)."""
    RegresionP1().ejecutar()
# ------------------------------------------------------


if __name__ == "__main__":
    app()