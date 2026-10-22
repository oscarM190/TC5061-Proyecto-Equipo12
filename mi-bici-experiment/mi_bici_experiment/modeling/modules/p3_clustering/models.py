"""P3: grupos de estaciones con patrones de uso similares y su ubicación en la ciudad.

Partición temporal (fechas en config.py):
  - Entrenamiento: P3_INICIO_ENTRENAMIENTO a P3_FIN_ENTRENAMIENTO.
  - Validación: el modelo se ajusta hasta 3 meses antes del fin del entrenamiento y se mide el ARI
    de estabilidad en esos 3 meses. Es el criterio para elegir hiperparámetros.
  - Prueba: el modelo de entrenamiento asigna el perfil de P3_FIN_ENTRENAMIENTO a FECHA_FIN y se
    mide el ARI contra los grupos de entrenamiento. Solo se reporta.

Cada corrida se lanza con `mlflow run` (ver MLproject y README).
"""

import json
import os
from pathlib import Path
import tempfile

import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici_experiment import plots
from mi_bici_experiment.config import (
    DESTINO_COL,
    EQUIPO,
    FECHA_FIN,
    FECHA_FIN_COL,
    FECHA_INICIO_COL,
    MLFLOW_EXPERIMENT,
    NOMENCLATURA_CSV,
    P3_FIN_ENTRENAMIENTO,
    P3_INICIO_ENTRENAMIENTO,
    P3_K_BASELINE,
    P3_K_CANDIDATO,
    P3_MESES_VALIDACION,
    P3_MIN_PROP_GRUPO,
    P3_MIN_VIAJES,
    P3_N_INIT,
    P3_PCA_COMPONENTES,
    PROJ_ROOT,
    RANDOM_STATE,
    STATION_COL,
    TRIPS_PARQUET,
)
from mi_bici_experiment.features import (
    columnas_perfil,
    filtrar_viajes_validos,
    info_estaciones,
    perfil_estaciones,
)
from mi_bici_experiment.modeling.modules.p3_clustering.estimadores import (
    AglomerativoConCentroides,
)
from mi_bici_experiment.tracking import MANIFEST_PATH, git_info, version_datos_manifiesto

MODELOS = ("kmeans", "gmm", "aglomerativo")
NOMBRE_REGISTRO = f"mibici-equipo{EQUIPO}-p3-grupos"
DESCRIPCION_REGISTRO = (
    "P3 MiBici: asigna cada estación a un grupo según su patrón de uso diario (peso del fin de "
    "semana, intensidad de uso y balance llegadas/salidas). Entrada: perfil de la estación "
    "(prop_fin_semana, log_viajes_dia, balance). Salida: número de grupo. Usuario: planeación "
    "urbana y expansión del sistema. Alias: champion (producción), challenger (siguiente mejor)."
)
TIPO_CONFIABLE = (
    "mi_bici_experiment.modeling.modules.p3_clustering.estimadores.AglomerativoConCentroides"
)

_DIA = pd.Timedelta(days=1)
INICIO_PRUEBA = str((pd.Timestamp(P3_FIN_ENTRENAMIENTO) + _DIA).date())
INICIO_VALIDACION = str(
    (pd.Timestamp(INICIO_PRUEBA) - pd.DateOffset(months=P3_MESES_VALIDACION)).date()
)
FIN_AJUSTE_VALIDACION = str((pd.Timestamp(INICIO_VALIDACION) - _DIA).date())


def construir(model: str, k: int, pca: int = 0, cov: str = "full", linkage: str = "ward",
              n_init: int = P3_N_INIT, seed: int = RANDOM_STATE) -> Pipeline:
    """Estandarización, PCA opcional y el algoritmo de clustering elegido."""
    pasos = [("escalado", StandardScaler())]
    if pca:
        pasos.append(("pca", PCA(n_components=pca, random_state=seed)))
    if model == "kmeans":
        pasos.append(("modelo", KMeans(n_clusters=k, n_init=n_init, random_state=seed)))
    elif model == "gmm":
        pasos.append(("modelo", GaussianMixture(n_components=k, covariance_type=cov,
                                                n_init=5, random_state=seed)))
    elif model == "aglomerativo":
        pasos.append(("modelo", AglomerativoConCentroides(n_clusters=k, linkage=linkage)))
    else:
        raise ValueError(f"Modelo P3 desconocido: {model}. Opciones: {', '.join(MODELOS)}")
    return Pipeline(pasos)


def etiquetas(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Grupos de las estaciones de ajuste (el aglomerativo los toma de su propio ajuste)."""
    if isinstance(pipe[-1], AglomerativoConCentroides):
        return pipe[-1].labels_
    return pipe.predict(X)


def ari_estabilidad(pipe: Pipeline, perfil: pd.DataFrame, perfil_posterior: pd.DataFrame,
                    cols: list[str]) -> float:
    """ARI entre los grupos de ajuste y los que asigna el modelo con el perfil posterior.

    Solo se comparan estaciones presentes en ambos periodos.
    """
    grupos = etiquetas(pipe, perfil[cols])
    comunes = perfil["estacion_id"].isin(perfil_posterior["estacion_id"]).to_numpy()
    posterior = perfil_posterior.set_index("estacion_id").loc[
        perfil.loc[comunes, "estacion_id"], cols
    ]
    return adjusted_rand_score(grupos[comunes], pipe.predict(posterior))


class DatosP3:
    """Perfiles por estación de entrenamiento, validación y prueba, construidos una sola vez."""

    def __init__(self, inicio: str = P3_INICIO_ENTRENAMIENTO):
        if not TRIPS_PARQUET.is_file():
            raise FileNotFoundError(
                f"No existe {TRIPS_PARQUET}; ejecuta primero `mlflow run . -e ingest`"
            )
        cols = [STATION_COL, DESTINO_COL, FECHA_INICIO_COL, FECHA_FIN_COL]
        viajes = filtrar_viajes_validos(pd.read_parquet(TRIPS_PARQUET, columns=cols))
        primer_dia = viajes[FECHA_INICIO_COL].min().normalize()
        if pd.Timestamp(inicio) < primer_dia:
            raise ValueError(
                f"inicio={inicio} es anterior a los datos ({primer_dia.date()}); "
                "ajusta P3_INICIO_ENTRENAMIENTO en config.py o el rango de MI_BICI_CSVS"
            )
        self.inicio = inicio
        self.train = perfil_estaciones(viajes, inicio, P3_FIN_ENTRENAMIENTO)
        self.test = perfil_estaciones(viajes, INICIO_PRUEBA, FECHA_FIN)
        self.ajuste_val = perfil_estaciones(viajes, inicio, FIN_AJUSTE_VALIDACION)
        self.val = perfil_estaciones(viajes, INICIO_VALIDACION, P3_FIN_ENTRENAMIENTO)
        self.cols = columnas_perfil(self.train)


def metricas(pipe: Pipeline, pipe_val: Pipeline, datos: DatosP3) -> dict:
    """Cohesión y separación en entrenamiento, estabilidad en validación y en prueba.

    Silueta y Davies-Bouldin se miden en el espacio común (las variables estandarizadas, sin
    PCA) para que todos los modelos sean comparables entre sí y con el baseline.
    """
    X = datos.train[datos.cols]
    grupos = etiquetas(pipe, X)
    X_comun = StandardScaler().fit_transform(X)
    tamanos = np.bincount(grupos)
    return {
        "train_silhouette_espacio_comun": silhouette_score(X_comun, grupos),
        "train_davies_bouldin_espacio_comun": davies_bouldin_score(X_comun, grupos),
        "train_calinski_harabasz_espacio_comun": calinski_harabasz_score(X_comun, grupos),
        "train_tamano_min_grupo": int(tamanos.min()),
        "train_prop_min_grupo": float(tamanos.min() / len(grupos)),
        "val_ari_estabilidad": ari_estabilidad(pipe_val, datos.ajuste_val, datos.val, datos.cols),
        "test_ari_estabilidad": ari_estabilidad(pipe, datos.train, datos.test, datos.cols),
    }


def resumen_grupos(estaciones: pd.DataFrame) -> pd.DataFrame:
    """Tamaño, municipios, centro y métricas de uso promedio por grupo."""
    g = estaciones.groupby("grupo")
    return pd.DataFrame(
        {
            "estaciones": g.size(),
            "municipios": g["municipio"].agg(
                lambda s: ", ".join(f"{m} {p:.0%}" for m, p in s.value_counts(True).head(3).items())
            ),
            "lat_centro": g["lat"].mean().round(4),
            "lon_centro": g["lon"].mean().round(4),
            "prop_fin_semana": g["prop_fin_semana"].mean().round(3),
            "viajes_dia": np.expm1(g["log_viajes_dia"].mean()).round(1),
            "balance": g["balance"].mean().round(3),
        }
    ).reset_index()


def _registrar_artefactos(pipe: Pipeline, datos: DatosP3, nombre: str, params: dict) -> None:
    """Modelo con firma, versión de los datos, configuración, gráficas y tablas en subcarpetas."""
    X = datos.train[datos.cols]
    mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="data")
    mlflow.log_input(
        mlflow.data.from_pandas(
            datos.train[["estacion_id", *datos.cols]], name=f"p3_perfil_train_{datos.inicio}"
        ),
        context="training",
    )
    for archivo in (PROJ_ROOT.parent / "MLproject", PROJ_ROOT.parent / "python_env.yaml"):
        if archivo.is_file():
            mlflow.log_artifact(str(archivo), artifact_path="config")
    with tempfile.TemporaryDirectory() as carpeta:
        carpeta = Path(carpeta)
        (carpeta / "params.json").write_text(json.dumps(params, indent=2, default=str))
        mlflow.log_artifact(str(carpeta / "params.json"), artifact_path="config")

        estaciones = datos.train.assign(grupo=etiquetas(pipe, X)).merge(
            info_estaciones(pd.read_csv(NOMENCLATURA_CSV)), on="estacion_id"
        )
        k = estaciones["grupo"].nunique()
        for figura in (
            plots.mapa_grupos(estaciones, f"P3: {k} grupos ({nombre})", carpeta / "mapa_grupos.png"),
            plots.perfil_grupos(estaciones, carpeta / "perfil_grupos.png"),
        ):
            mlflow.log_artifact(str(figura), artifact_path="plots")
        tablas = {
            "estaciones_por_grupo.csv": estaciones[
                ["estacion_id", "nombre", "municipio", "lat", "lon", "capacidad", "grupo"]
            ],
            "resumen_grupos.csv": resumen_grupos(estaciones),
        }
        for archivo, tabla in tablas.items():
            tabla.to_csv(carpeta / archivo, index=False)
            mlflow.log_artifact(str(carpeta / archivo), artifact_path="tables")

    ejemplo = X.head(5).astype("float64")
    mlflow.sklearn.log_model(
        pipe,
        name="model",
        signature=infer_signature(ejemplo, pipe.predict(ejemplo)),
        input_example=ejemplo,
        skops_trusted_types=[TIPO_CONFIABLE] if isinstance(pipe[-1], AglomerativoConCentroides)
        else None,
    )


def entrenar(datos: DatosP3, model: str = "kmeans", k: int = 3, pca: int = 0, cov: str = "full",
             linkage: str = "ward", n_init: int = P3_N_INIT, seed: int = RANDOM_STATE,
             stage: str = "candidate", run_name: str | None = None, nested: bool = False,
             artefactos: bool = True) -> tuple[str, dict]:
    """Ajusta un modelo P3 y registra parámetros, métricas, etiquetas y artefactos en MLflow.

    Los parámetros se registran con los mismos nombres que `mlflow run -P`, salvo la semilla
    (`random_state`) y la fecha (`inicio_entrenamiento`), que pueden venir vacías desde el
    MLproject y resolverse en config.py.
    """
    hp = {"model": model, "k": k, "pca": pca, "cov": cov, "linkage": linkage, "n_init": n_init}
    nombre = run_name or f"p3-{model}-k{k}" + (f"-pca{pca}" if pca else "")
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.set_tags(
            {
                "equipo": EQUIPO,
                "pregunta": "P3",
                "question": "p3",
                "model_family": model,
                "stage_of_work": stage,
                "entorno": os.getenv("MIBICI_ENTORNO", "local"),
                **git_info(),
            }
        )
        params = {
            **hp,
            "random_state": seed,
            "inicio_entrenamiento": datos.inicio,
            "fin_entrenamiento": P3_FIN_ENTRENAMIENTO,
            "validacion": f"{INICIO_VALIDACION}..{P3_FIN_ENTRENAMIENTO}",
            "prueba": f"{INICIO_PRUEBA}..{FECHA_FIN}",
            "data_version": version_datos_manifiesto(),
            "min_viajes": P3_MIN_VIAJES,
            "n_estaciones": len(datos.train),
            "features": ",".join(datos.cols),
        }
        mlflow.log_params(params)
        pipe = construir(model, k, pca, cov, linkage, n_init, seed).fit(datos.train[datos.cols])
        pipe_val = construir(model, k, pca, cov, linkage, n_init, seed).fit(
            datos.ajuste_val[datos.cols]
        )
        resultado = metricas(pipe, pipe_val, datos)
        mlflow.log_metrics(resultado)
        if artefactos:
            _registrar_artefactos(pipe, datos, nombre, params)
        return run.info.run_id, resultado


def buscar(datos: DatosP3, model: str, ks: list[int], pcas: list[int], seed: int = RANDOM_STATE):
    """Búsqueda organizada como nested runs: una corrida padre y una hija por combinación.

    Se elige la combinación con mayor ARI de validación entre las que dejan todos los grupos
    con al menos P3_MIN_PROP_GRUPO de las estaciones; la prueba no interviene.
    """
    with mlflow.start_run(run_name=f"p3-search-{model}") as padre:
        mlflow.set_tags({"equipo": EQUIPO, "pregunta": "P3", "question": "p3",
                         "model_family": model, "stage_of_work": "search", **git_info()})
        filas = []
        for k in ks:
            for pca in pcas:
                _, m = entrenar(datos, model, k, pca, seed=seed, stage="search", nested=True,
                                run_name=f"p3-search-{model}-k{k}-pca{pca}", artefactos=False)
                filas.append({"k": k, "pca": pca, **m})
        tabla = pd.DataFrame(filas)
        validas = tabla[tabla["train_prop_min_grupo"] >= P3_MIN_PROP_GRUPO]
        mejor = validas.sort_values(
            ["val_ari_estabilidad", "train_silhouette_espacio_comun"], ascending=False
        ).iloc[0]
        mlflow.log_params({
            "k_candidatos": ",".join(map(str, ks)),
            "pca_candidatos": ",".join(map(str, pcas)),
            "criterio": f"max val_ari_estabilidad con grupos >= {P3_MIN_PROP_GRUPO:.0%}",
            "mejor_k": int(mejor["k"]),
            "mejor_pca": int(mejor["pca"]),
        })
        mlflow.log_metrics({"mejor_val_ari_estabilidad": mejor["val_ari_estabilidad"],
                            "mejor_train_silhouette_espacio_comun":
                                mejor["train_silhouette_espacio_comun"]})
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "busqueda.csv"
            tabla.to_csv(ruta, index=False)
            mlflow.log_artifact(str(ruta), artifact_path="tables")
            for pca in pcas:
                sub = tabla[tabla["pca"] == pca]
                fig = plots.silueta_por_k(sub["k"].tolist(),
                                          sub["train_silhouette_espacio_comun"].tolist(),
                                          Path(carpeta) / f"silueta_por_k_pca{pca}.png")
                mlflow.log_artifact(str(fig), artifact_path="plots")
        return padre.info.run_id, int(mejor["k"]), int(mejor["pca"])


def registrar(run_id: str, alias: str = "champion"):
    """Registra el modelo de una corrida, conserva su linaje y le asigna un alias.

    Devuelve la versión y el modelo cargado por `models:/<nombre>@<alias>`.
    """
    cliente = mlflow.MlflowClient()
    if not cliente.search_registered_models(filter_string=f"name = '{NOMBRE_REGISTRO}'"):
        cliente.create_registered_model(NOMBRE_REGISTRO, description=DESCRIPCION_REGISTRO,
                                        tags={"equipo": EQUIPO, "question": "p3"})
    origen = cliente.get_run(run_id)
    version = mlflow.register_model(
        f"runs:/{run_id}/model",
        NOMBRE_REGISTRO,
        tags={
            "experimento": MLFLOW_EXPERIMENT,
            "run_origen": run_id,
            "git_commit": origen.data.tags.get("git_commit", ""),
            "data_version": origen.data.params.get("data_version", ""),
        },
    )
    cliente.update_model_version(
        NOMBRE_REGISTRO,
        version.version,
        description=(
            f"{origen.info.run_name}: val_ari_estabilidad="
            f"{origen.data.metrics['val_ari_estabilidad']:.3f}, test_ari_estabilidad="
            f"{origen.data.metrics['test_ari_estabilidad']:.3f}, train_silhouette_espacio_comun="
            f"{origen.data.metrics['train_silhouette_espacio_comun']:.3f}"
        ),
    )
    cliente.set_registered_model_alias(NOMBRE_REGISTRO, alias, version.version)
    return version.version, mlflow.pyfunc.load_model(f"models:/{NOMBRE_REGISTRO}@{alias}")


class ClusteringModel:
    """Compatibilidad con `python -m mi_bici_experiment.modeling.train`: baseline y candidato."""

    def ejecutar(self):
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        datos = DatosP3()
        entrenar(datos, "kmeans", P3_K_BASELINE, stage="baseline")
        entrenar(datos, "kmeans", P3_K_CANDIDATO, P3_PCA_COMPONENTES)
