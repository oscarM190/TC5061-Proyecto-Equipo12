import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature

from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mi_bici_experiment import config
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
    P1_COMBINACIONES,
    PM_DIAS,
    PROJ_ROOT,
    RANDOM_STATE,
    STATION_COL,
    TEST_PARTICION,
    VAL_PARTICION,
    P1_NOMBRE_REGISTRO
)
from mi_bici_experiment.features import (
    agregar_variables,
    construir_viajes_diarios,
    limpiar_viajes,
)
from mi_bici_experiment.tracking import git_info, version_datos

MANIFIESTO = PROJ_ROOT / "data_manifest.json"


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
                "model_family": "linear",
                "tipo_modelo": "baseline",
                **git_info(),
                "entorno": os.getenv("MIBICI_ENTORNO", "local"),
            }):
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


# ----------------------------------------------------------------------.
# Entrega 2: LightGBM con búsqueda de hiperparámetros (nested runs)
# ---------------------------------------------------------------------
class RegressionLGBM(RegressionModel):
    """Misma preparación de datos y partición que el baseline, solo cambia el modelo."""
    hiperparametros:dict ={}

    def construir_modelo(self):
        pipeline = super().construir_modelo()
        pipeline.set_params(
            modelo=LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **self.hiperparametros))
        return pipeline

    def registrar_corrida(self):
        """Parámetros, métricas, gráficas, datos, configuración y modelo de una run candidata."""
        X = self.prueba[self.todas_var]
        mlflow.log_params(
            {
                "fecha_inicio": self.fecha_inicio,
                "fecha_fin": self.fecha_fin,
                "fraccion_validacion": self.fraccion_val,
                "fraccion_test": self.fraccion_test,
                "data_version": version_datos(),
                "random_state": RANDOM_STATE,
                "modelo": "LGBMRegressor",
                **self.hiperparametros
            })
        mlflow.log_metrics(self.metricas)

        # Datos y configuración
        entrenamiento = self.entrenamiento[self.todas_var + [OBJETIVO_COL]]
        mlflow.log_input(mlflow.data.from_pandas(entrenamiento, name="train_p1"), context="training")
        mlflow.log_artifact(str(MANIFIESTO), artifact_path="data")
        mlflow.log_artifact(config.__file__, artifact_path="config")

        # Gráficas real vs predicho y residuales (test)
        por_dia = (
            self.prueba[["date"]]
            .assign(real=self.prueba[OBJETIVO_COL].to_numpy(), predicho=self.y_pred)
            .groupby("date")
            .sum()
        )
        fig, ax = plt.subplots()
        por_dia.plot(ax=ax, title="Viajes diarios totales")
        mlflow.log_figure(fig, "plots/real_vs_predicho.png")
        plt.close(fig)

        residuales = self.prueba[OBJETIVO_COL].to_numpy()-self.y_pred
        fig, ax = plt.subplots()
        ax.hist(residuales, bins=50)
        ax.set_title("Residuales (con test para P1)")
        mlflow.log_figure(fig, "plots/residuales.png")
        plt.close(fig)

        # Modelo con firma y ejemplo de entrada
        mlflow.sklearn.log_model(
            self.modelo,
            artifact_path="model",
            signature=infer_signature(X, self.y_pred),
            input_example=X.head(5),
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRegressor"
            ])

# ----------------------------------
# Model Registry
# ---------------------------------
def promover_a_produccion(run_id: str):
    """Registra el mejor modelo, le pone descripción y su linaje, y le asigna el alias "champion"."""
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/model", P1_NOMBRE_REGISTRO).version

    client.update_registered_model(
        P1_NOMBRE_REGISTRO,
        description="P1:predicción de viajes diarios por estación usando el modelo LightGBM.")
    client.update_model_version(
        P1_NOMBRE_REGISTRO, version, description=f"Mejor corrida por val_mae (run {run_id}).")

    for clave, valor in {**git_info(), "data_version": version_datos()}.items():
        client.set_model_version_tag(P1_NOMBRE_REGISTRO, version, clave, str(valor))
    client.set_registered_model_alias(P1_NOMBRE_REGISTRO, "champion",version)

    mlflow.sklearn.load_model(f"models:/{P1_NOMBRE_REGISTRO}@champion")
    print(f"{P1_NOMBRE_REGISTRO} v{version}cargado con alias champion")


    
def buscar_hiperparametros(combinaciones=None, fecha_inicio=FECHA_INICIO, fecha_fin=FECHA_FIN):
    """Baseline con el run padre el cual contiene un hijo por combinación y gráfica de comparación como artefacto."""
    combinaciones= combinaciones or P1_COMBINACIONES
    modelo = RegressionLGBM(fecha_inicio, fecha_fin)
    modelo.preparar_datos()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Baseline (regresión lineal), como otra run independiente
    baseline = RegressionModel(fecha_inicio, fecha_fin)
    baseline.entrenamiento = modelo.entrenamiento
    baseline.validacion = modelo.validacion
    baseline.prueba = modelo.prueba
    baseline.entrenar()
    metricas_base = dict(baseline.evaluar())
    baseline.registrar_mlflow()

    etiquetas = {
        "equipo": EQUIPO,
        "pregunta": "P1",
        "model_family": "lightgbm",
        **git_info(),
        "entorno": os.getenv("MIBICI_ENTORNO", "local")}


    resultados = []
    with mlflow.start_run():  # run padre, la búsqueda del mejor modelo
        mlflow.set_tag("mlflow.runName", "P1-lightgbm-final")
        mlflow.set_tags({**etiquetas, "stage_of_work": "search"})

        for nombre, hiperparametros in combinaciones.items():
            modelo.hiperparametros = hiperparametros
            modelo.entrenar()
            modelo.evaluar()
            with mlflow.start_run(run_name=f"P1-lightgbm-{nombre}", nested=True) as hijo:
                mlflow.set_tags({**etiquetas, "stage_of_work": "candidate"})
                modelo.registrar_corrida()
            resultados.append((modelo.metricas["val_mae"], hijo.info.run_id, nombre, dict(modelo.metricas)))

        mejor_mae, mejor_run, mejor_nombre,metricas_mejor= min(resultados, key=lambda r:r[0])

        # Se busca la mejor run o combinación y se registran sus métricas y parámetros en el run padre
        mlflow.log_params({"mejor_combinacion": mejor_nombre, **combinaciones[mejor_nombre]})
        mlflow.log_metrics({f"best_{k}": v for k, v in metricas_mejor.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in metricas_base.items()})
        for k in ("val_mae", "test_mae"):
            mejora = (metricas_base[k] - metricas_mejor[k]) /metricas_base[k]*100
            mlflow.log_metric(f"mejora_{k}_pct", mejora)
        mlflow.set_tag("best_run_id", mejor_run)

        # Gráfica baseline vs mejor modelo
        nombres = ["val_mae", "test_mae", "val_rmse", "test_rmse"]
        x = np.arange(len(nombres))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - 0.2, [metricas_base[n] for n in nombres], 0.4, label="Baseline (regresión lineal)")
        ax.bar(x + 0.2, [metricas_mejor[n] for n in nombres], 0.4, label=f"LightGBM ({mejor_nombre})")
        ax.set_xticks(x)
        ax.set_xticklabels(nombres)
        ax.set_ylabel("Error (viajes por estación y día)")
        ax.set_title("P1: baseline vs mejor modelo (menor valor es mejor modelo)")
        ax.legend()
        mlflow.log_figure(fig, "plots/baseline_vs_mejor.png")
        plt.close(fig)

    promover_a_produccion(mejor_run)


app = typer.Typer()

@app.command()
def main(
    n_estimators: int | None = None,
    learning_rate: float | None = None,
    num_leaves: int | None = None,
    min_child_samples: int | None = None,
    fecha_inicio: str = FECHA_INICIO,
    fecha_fin: str = FECHA_FIN):
    """Sin hiperparámetros corre la búsqueda con los valores que hay en config.py, 
    con algún parámetro, entrena esa combinación nada más."""
    elegidos = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples}
    elegidos = {k: v for k, v in elegidos.items() if v is not None}
    combinaciones = None
    if elegidos:
        # Parte de la combinación del "modelo_3" de config.py
        combinaciones = {"personalizada": {**P1_COMBINACIONES["modelo_3"], **elegidos}}
    buscar_hiperparametros(combinaciones, fecha_inicio, fecha_fin)


if __name__ == "__main__":
    app()