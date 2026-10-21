"""P3 station clustering training pipeline."""

import os

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici_experiment import plots
from mi_bici_experiment.config import (
    DESTINO_COL,
    EQUIPO,
    FECHA_FIN,
    FECHA_FIN_COL,
    FECHA_INICIO,
    FECHA_INICIO_COL,
    TRIPS_PARQUET,
    MLFLOW_EXPERIMENT,
    NOMENCLATURA_CSV,
    P3_FIGURES_DIR,
    P3_FIN_ENTRENAMIENTO,
    P3_K_BASELINE,
    P3_K_CANDIDATOS,
    P3_MIN_VIAJES,
    P3_N_INIT,
    P3_PCA_COMPONENTES,
    RANDOM_STATE,
    STATION_COL,
)
from mi_bici_experiment.features import (
    columnas_perfil,
    filtrar_viajes_validos,
    info_estaciones,
    perfil_estaciones,
)
from mi_bici_experiment.tracking import git_info, version_datos


class ClusteringModel:
    """P3: grupos de estaciones con métricas de uso similares y su ubicación en la ciudad.

    Baseline: k-means estandarizado con k fijo sobre tres métricas de uso diario.
    Candidato: estandarización, PCA y k-means sobre las mismas métricas, con k elegido por silueta.
    Métricas: silueta, Davies-Bouldin y Calinski-Harabasz en entrenamiento, y ARI de estabilidad
    entre los grupos de entrenamiento y los que se asignan con el perfil del periodo posterior.
    Las métricas `*_espacio_comun` se calculan para ambos modelos sobre las tres métricas
    estandarizadas, el espacio en el que baseline y candidato se pueden comparar.
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
        viajes = filtrar_viajes_validos(pd.read_parquet(TRIPS_PARQUET, columns=cols))
        self.perfil = perfil_estaciones(viajes, FECHA_INICIO, self.fin_entrenamiento)
        self.perfil_test = perfil_estaciones(viajes, self.inicio_test, FECHA_FIN)
        self.info = info_estaciones(pd.read_csv(NOMENCLATURA_CSV))
        self.data_version = version_datos()
        logger.info(
            f"P3: {len(self.perfil)} estaciones en entrenamiento, "
            f"{len(self.perfil_test)} en el periodo de estabilidad"
        )

    def ajustar(self, cols: list[str], k: int, reducir: bool):
        """Estandariza las métricas y aplica PCA opcional antes de K-means."""
        pasos = [StandardScaler()]
        if reducir:
            pasos.append(PCA(P3_PCA_COMPONENTES, random_state=RANDOM_STATE))
        pipe = make_pipeline(
            *pasos,
            KMeans(n_clusters=k, n_init=P3_N_INIT, random_state=RANDOM_STATE),
        )
        grupos = pipe.fit_predict(self.perfil[cols])
        X = pipe[:-1].transform(self.perfil[cols])
        return pipe, grupos, X

    def metricas(self, pipe, grupos, X, cols) -> dict:
        """Cohesión y separación en entrenamiento y estabilidad en el periodo posterior.

        Las métricas train se miden en el espacio ajustado (métricas estandarizadas en el
        baseline, PCA en el candidato). Las versiones `_espacio_comun` comparan ambos grupos
        sobre las tres métricas estandarizadas.
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
        """Tamaño, municipios, centro y métricas de uso promedio por grupo."""
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
                    "escalado": True,
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

        cols = columnas_perfil(self.perfil)
        self.registrar(
            f"P3-kmeans-baseline-k{P3_K_BASELINE}",
            "baseline",
            "K-means estandarizado con k fijo sobre tres métricas de uso de cada estación.",
            cols,
            P3_K_BASELINE,
            reducir=False,
            params={"features": "uso de fin de semana, intensidad diaria y balance"},
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
            "Estandarización, PCA y k-means sobre uso de fin de semana, intensidad diaria y "
            "balance; k elegido por silueta.",
            cols,
            k_opt,
            reducir=True,
            params={
                "features": "uso de fin de semana, intensidad diaria y balance",
                "k_candidatos": ",".join(map(str, P3_K_CANDIDATOS)),
            },
            extra_figs=[fig_k],
        )
