from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mi_bici_experiment.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from mi_bici_experiment.plots import matriz_confucion_regresion_multiclass_clasificacion

import os
import json
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from mi_bici_experiment.features import agregar_balance_neto_y_normalizado, clasificacion_variable_objetivo_y_caracterizticas, creacion_variables_memoria_historica, creacion_variables_geoespaciales


from mi_bici_experiment import plots
from mi_bici_experiment.config import (
    DESTINO_COL,
    EQUIPO,
    FECHA_FIN,
    FECHA_FIN_COL,
    FECHA_INICIO,
    FECHA_INICIO_COL,
    FEATURES_PARQUET,
    FIGURES_DIR,
    LAGS,
    MLFLOW_EXPERIMENT,
    NOMENCLATURA_PARQUET,
    OBJETIVO_COL,
    P3_FIGURES_DIR,
    P3_FIN_ENTRENAMIENTO,
    P3_K_BASELINE,
    P3_K_CANDIDATOS,
    P3_MIN_VIAJES,
    P3_N_INIT,
    P3_PCA_COMPONENTES,
    PM_DIAS,
    RANDOM_STATE,
    STATION_COL,
    TEST_PARTICION,
)
from mi_bici_experiment.features import (
    agregar_variables,
    columnas_perfil,
    construir_viajes_diarios,
    filtrar_viajes_validos,
    info_estaciones,
    limpiar_viajes,
    perfil_estaciones,
)
from mi_bici_experiment.tracking import git_info, version_datos

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

# ---------------------------------------------------------------------------------------------
# P3 (Clasificacion): Estaciones en riesgo de quedarse sin bicis.
# ---------------------------------------------------------------------------------------------
class RegresionLinealMultinomialP2:
    """Entrena, evalúa y registra el clasificador multinomial en MLflow."""

    def __init__(self, dataset_ml: pd.DataFrame):
        self.dataset_ml = dataset_ml
        self.features = [
            'dpcapacity', 'Dia_Semana', 'Es_Fin_Semana', 'Mes', 
            'Balance_Lag_1', 'Balance_Lag_2', 'Balance_Lag_7',
            'Estado_Lag_1', 'Estado_Lag_7', 
            'Media_Movil_3d', 'Media_Movil_7d', 'Volatilidad_3d',
            'Zona_Espacial', 'Densidad_500m'
        ]

        # Construcción del Pipeline de Machine Learning
        # El pipeline garantiza que la estandarización se ajuste solo con los datos de entrenamiento
        self.pipeline_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('clasificador', LogisticRegression(
                max_iter=1000, # Incrementado para asegurar convergencia con múltiples variables
                class_weight='balanced', # Crucial si la clase 1 (balanceado) domina la distribución
                random_state=RANDOM_STATE
            ))
        ])

    def entrenar(self):
        """Separa los datos 70/15/15 y registra el entrenamiento y sus resultados."""
        X = self.dataset_ml[self.features]
        y = self.dataset_ml['Riesgo_T1']

        # Separación cronológica: 70 % entrenamiento, 15 % validación y 15 % prueba.
        # El dataset debe estar ordenado por fecha antes de llamar a esta función.
        X_train, X_remaining, y_train, y_remaining = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_remaining, y_remaining, test_size=0.5, shuffle=False
        )

        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        tipo = "regresion-lineal-multinomial-clasificacion"
        descripcion = (
            "P2: regresión logística multinomial con escalado; separación cronológica "
            "70 % entrenamiento, 15 % validación y 15 % prueba. Métricas globales macro."
        )
        self.data_version = version_datos()
        with mlflow.start_run(run_name=tipo, tags={
            "equipo": EQUIPO,
            "pregunta": "P2",
            "tipo_modelo": tipo,
            **git_info(),
            "entorno": os.getenv("MIBICI_ENTORNO", "local"),
            "mlflow.note.content": descripcion,
        }):
            params = {
                key: value for key, value in self.pipeline_lr.get_params(deep=True).items()
                if key.startswith(("scaler__", "clasificador__"))
            }
            params.update({
                "train_fraccion": 0.70, "validacion_fraccion": 0.15, "test_fraccion": 0.15,
                "shuffle": False, "version_datos": self.data_version,
                "features": json.dumps(self.features), "objetivo": "Riesgo_T1",
                "promedio_metricas": "macro", "n_train": len(X_train),
                "n_validacion": len(X_val), "n_test": len(X_test), "fecha_inicio": FECHA_INICIO, "fecha_fin": FECHA_FIN, "random_state": RANDOM_STATE
            })
            mlflow.log_params(params)
            # Entrenamiento del Modelo
            self.pipeline_lr.fit(X_train, y_train)

            # Evaluación sin volver a ajustar el modelo ni el escalador.
            for name, X_eval, y_eval in (
                ('Validación', X_val, y_val),
                ('Prueba', X_test, y_test),
            ):
                y_pred = self.pipeline_lr.predict(X_eval)
                print(f"Reporte de Clasificación - {name} (Baseline Multinomial):")
                print(classification_report(
                    y_eval, y_pred,
                    labels=[0, 1, 2],
                    target_names=['Sin Bicis (0)', 'Balanceado (1)', 'Sin Anclajes (2)'],
                    zero_division=0,
                ))
                report = classification_report(
                    y_eval, y_pred, labels=[0, 1, 2], output_dict=True, zero_division=0
                )
                prefix = "val" if name == "Validación" else "test"
                mlflow.log_metrics({
                    f"{prefix}_{metric}": report["macro avg"][metric]
                    for metric in ("precision", "recall", "f1-score")
                })
                for label in ("0", "1", "2"):
                    mlflow.log_metrics({
                        f"{prefix}_clase_{label}_{metric}": report[label][metric]
                        for metric in ("precision", "recall", "f1-score")
                    })
                matrix_path = matriz_confucion_regresion_multiclass_clasificacion(
                    y_eval, y_pred, conjunto=name
                )
                mlflow.log_artifact(str(matrix_path), artifact_path="figures")
            mlflow.log_artifact(str(FIGURES_DIR / "balance_normalizado.png"), artifact_path="figures")
            input_example = X_train.head(5)
            mlflow.sklearn.log_model(
                self.pipeline_lr,
                name=tipo,
                registered_model_name=tipo,
                signature=infer_signature(input_example, self.pipeline_lr.predict(input_example)),
                input_example=input_example,
            )
        return self.pipeline_lr


    @staticmethod
    def ejecutar():
        df = pd.read_parquet(INTERIM_DATA_DIR / 'features.parquet')
        balance_diario = agregar_balance_neto_y_normalizado(df)
        dataset_ml = clasificacion_variable_objetivo_y_caracterizticas(balance_diario)
        dataset_ml = creacion_variables_memoria_historica(dataset_ml)
        dataset_ml = creacion_variables_geoespaciales(dataset_ml, df)
        modelo = RegresionLinealMultinomialP2(dataset_ml)
        modelo.entrenar()


@app.command()
def clasificacion_p2():
    """Ejecuta el baseline de regresión logistica multinomial (P2)."""
    RegresionLinealMultinomialP2().ejecutar()


# ---------------------------------------------------------------------------------------------
# P3 (clustering): grupos de estaciones que comparten patrones de uso
# ---------------------------------------------------------------------------------------------


class ClusteringP3:
    """P3: grupos de estaciones con patrones de uso similares y su ubicación en la ciudad.

    Baseline: k-means con k fijo sobre el perfil horario de salidas en días hábiles.
    Candidato: estandarización, PCA y k-means sobre el perfil completo, con k elegido por silueta.
    Métricas: silueta, Davies-Bouldin y Calinski-Harabasz en entrenamiento, y ARI de estabilidad
    entre los grupos de entrenamiento y los que se asignan con el perfil del periodo posterior.
    Las métricas `*_espacio_comun` se calculan para ambos modelos sobre el perfil completo
    estandarizado, el único espacio en el que baseline y candidato se pueden comparar.
    """

    def __init__(self, fin_entrenamiento=P3_FIN_ENTRENAMIENTO):
        self.fin_entrenamiento = fin_entrenamiento
        self.inicio_test = str((pd.Timestamp(fin_entrenamiento) + pd.Timedelta(days=1)).date())
        self.perfil = None
        self.perfil_test = None
        self.info = None
        self.data_version = None

    def preparar_datos(self):
        """Limpia los viajes y construye el perfil de entrenamiento y el del periodo posterior."""
        cols = [STATION_COL, DESTINO_COL, FECHA_INICIO_COL, FECHA_FIN_COL]
        viajes = filtrar_viajes_validos(pd.read_parquet(FEATURES_PARQUET, columns=cols))
        self.perfil = perfil_estaciones(viajes, FECHA_INICIO, self.fin_entrenamiento)
        self.perfil_test = perfil_estaciones(viajes, self.inicio_test, FECHA_FIN)
        self.info = info_estaciones(pd.read_parquet(NOMENCLATURA_PARQUET))
        self.data_version = version_datos()
        logger.info(
            f"P3: {len(self.perfil)} estaciones en entrenamiento, "
            f"{len(self.perfil_test)} en el periodo de estabilidad"
        )

    def ajustar(self, cols: list[str], k: int, reducir: bool):
        """K-means directo o tras estandarizar y reducir con PCA.

        Las 48 proporciones horarias están correlacionadas; los primeros componentes resumen
        los picos de mañana y tarde y hacen que las distancias de k-means tengan sentido.
        """
        pasos = [StandardScaler(), PCA(P3_PCA_COMPONENTES, random_state=RANDOM_STATE)]
        pipe = make_pipeline(
            *(pasos if reducir else []), KMeans(n_clusters=k, n_init=P3_N_INIT, random_state=RANDOM_STATE)
        )
        grupos = pipe.fit_predict(self.perfil[cols])
        X = pipe[:-1].transform(self.perfil[cols]) if reducir else self.perfil[cols].to_numpy()
        return pipe, grupos, X

    def metricas(self, pipe, grupos, X, cols) -> dict:
        """Cohesión y separación en entrenamiento y estabilidad en el periodo posterior.

        `train_silhouette`, `train_davies_bouldin` y `train_calinski_harabasz` se miden en el
        espacio en que se ajustó cada modelo (perfil crudo en el baseline, PCA en el candidato),
        así que no son comparables entre runs. Las versiones `_espacio_comun` miden los mismos
        grupos sobre el perfil completo estandarizado y sí permiten compararlos.
        """
        X_comun = StandardScaler().fit_transform(self.perfil[columnas_perfil(self.perfil)])
        comunes = self.perfil["estacion_id"].isin(self.perfil_test["estacion_id"]).to_numpy()
        test = self.perfil_test.set_index("estacion_id").loc[
            self.perfil.loc[comunes, "estacion_id"]
        ]
        return {
            "train_silhouette": silhouette_score(X, grupos, random_state=RANDOM_STATE),
            "train_davies_bouldin": davies_bouldin_score(X, grupos),
            "train_calinski_harabasz": calinski_harabasz_score(X, grupos),
            "train_silhouette_espacio_comun": silhouette_score(
                X_comun, grupos, random_state=RANDOM_STATE
            ),
            "train_davies_bouldin_espacio_comun": davies_bouldin_score(X_comun, grupos),
            "train_tamano_min_grupo": int(np.bincount(grupos).min()),
            "test_ari_estabilidad": adjusted_rand_score(grupos[comunes], pipe.predict(test[cols])),
        }

    def resumen_grupos(self, estaciones: pd.DataFrame) -> pd.DataFrame:
        """Tamaño, municipios, centro, horas pico, fin de semana, uso y balance por grupo."""
        sal = columnas_perfil(estaciones, "sal_h")
        lle = columnas_perfil(estaciones, "lle_h")
        g = estaciones.groupby("grupo")
        return pd.DataFrame(
            {
                "estaciones": g.size(),
                "municipios": g["municipio"].agg(
                    lambda s: ", ".join(
                        f"{m} {p:.0%}" for m, p in s.value_counts(True).head(3).items()
                    )
                ),
                "lat_centro": g["lat"].mean().round(4),
                "lon_centro": g["lon"].mean().round(4),
                "hora_pico_salidas": g[sal].mean().idxmax(axis=1).str[-2:].astype(int),
                "hora_pico_llegadas": g[lle].mean().idxmax(axis=1).str[-2:].astype(int),
                "prop_fin_semana": g["prop_fin_semana"].mean().round(3),
                "viajes_dia": np.expm1(g["log_viajes_dia"].mean()).round(1),
                "balance": g["balance"].mean().round(3),
            }
        ).reset_index()

    def registrar(self, run_name, tipo, descripcion, cols, k, reducir, params, extra_figs=()):
        """Ajusta un modelo y registra parámetros, métricas, etiquetas y artefactos en MLflow."""
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "equipo": EQUIPO,
                    "pregunta": "P3",
                    "tipo_modelo": tipo,
                    "entorno": os.getenv("MIBICI_ENTORNO", "local"),
                    **git_info(),
                    "mlflow.note.content": descripcion,
                }
            )
            pipe, grupos, X = self.ajustar(cols, k, reducir)
            mlflow.log_params(
                {
                    "fecha_inicio": FECHA_INICIO,
                    "fin_entrenamiento": self.fin_entrenamiento,
                    "inicio_estabilidad": self.inicio_test,
                    "fecha_fin": FECHA_FIN,
                    "data_version": self.data_version,
                    "random_state": RANDOM_STATE,
                    "min_viajes": P3_MIN_VIAJES,
                    "n_estaciones": len(self.perfil),
                    "modelo": "KMeans",
                    "n_clusters": k,
                    "n_init": P3_N_INIT,
                    "escalado": reducir,
                    "pca_componentes": P3_PCA_COMPONENTES if reducir else 0,
                    **params,
                }
            )
            metr = self.metricas(pipe, grupos, X, cols)
            mlflow.log_metrics(metr)

            estaciones = self.perfil.assign(grupo=grupos).merge(self.info, on="estacion_id")
            carpeta = P3_FIGURES_DIR / run_name
            figs = [
                plots.mapa_grupos(
                    estaciones, f"P3: {k} grupos ({tipo})", carpeta / "mapa_grupos.png"
                ),
                plots.perfil_grupos(estaciones, carpeta / "perfil_grupos.png"),
                *extra_figs,
            ]
            for f in figs:
                mlflow.log_artifact(str(f), artifact_path="figuras")
            tablas = {
                "estaciones_por_grupo.csv": estaciones[
                    ["estacion_id", "nombre", "municipio", "lat", "lon", "capacidad", "grupo"]
                ],
                "resumen_grupos.csv": self.resumen_grupos(estaciones),
            }
            for nombre, tabla in tablas.items():
                ruta = carpeta / nombre
                tabla.to_csv(ruta, index=False)
                mlflow.log_artifact(str(ruta), artifact_path="tablas")
            if reducir:
                mlflow.sklearn.log_model(
                    pipe, name="modelo", input_example=self.perfil[cols].head(5).astype("float64")
                )
            logger.info(f"{run_name}: k={k} train_silhouette={metr['train_silhouette']:.3f}")
            return metr

    def ejecutar(self):
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        self.preparar_datos()

        cols = columnas_perfil(self.perfil, "sal_h")
        self.registrar(
            f"P3-kmeans-baseline-k{P3_K_BASELINE}",
            "baseline",
            "K-means con k fijo sobre el perfil horario de salidas en días hábiles.",
            cols,
            P3_K_BASELINE,
            reducir=False,
            params={"features": "perfil horario de salidas (24)"},
        )

        cols = columnas_perfil(self.perfil)
        siluetas = []
        for k in P3_K_CANDIDATOS:
            _, grupos, X = self.ajustar(cols, k, reducir=True)
            siluetas.append(silhouette_score(X, grupos, random_state=RANDOM_STATE))
        k_opt = P3_K_CANDIDATOS[int(np.argmax(siluetas))]
        fig_k = plots.silueta_por_k(
            P3_K_CANDIDATOS,
            siluetas,
            P3_FIGURES_DIR / "P3-pca-kmeans-candidato" / "silueta_por_k.png",
        )
        self.registrar(
            "P3-pca-kmeans-candidato",
            "candidato",
            "Estandarización, PCA y k-means sobre perfiles de salidas y llegadas, fin de semana, "
            "uso y balance; k elegido por silueta.",
            cols,
            k_opt,
            reducir=True,
            params={
                "features": f"perfil completo ({len(cols)})",
                "k_candidatos": ",".join(map(str, P3_K_CANDIDATOS)),
            },
            extra_figs=[fig_k],
        )


@app.command()
def clustering_p3():
    """Ejecuta el baseline y el candidato de P1, P2 yP3 y los registra en MLflow."""
    RegresionP1.ejecutar()
    RegresionLinealMultinomialP2.ejecutar()
    ClusteringP3().ejecutar()

@app.command()
def todos():
    """Ejecuta P1, P2 y P3 en secuencia."""
    RegresionP1().ejecutar()
    RegresionLinealMultinomialP2.ejecutar()
    ClusteringP3().ejecutar()


if __name__ == "__main__":
    app()
