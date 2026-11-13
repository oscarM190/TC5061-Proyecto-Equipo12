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
    leer_nomenclatura,
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


def periodos(fin: str = P3_FIN_ENTRENAMIENTO) -> dict[str, str]:
    """Fechas de validación y prueba que se derivan del fin del entrenamiento."""
    inicio_prueba = pd.Timestamp(fin) + _DIA
    inicio_validacion = inicio_prueba - pd.DateOffset(months=P3_MESES_VALIDACION)
    return {
        "inicio_prueba": str(inicio_prueba.date()),
        "inicio_validacion": str(inicio_validacion.date()),
        "fin_ajuste_validacion": str((inicio_validacion - _DIA).date()),
    }


INICIO_PRUEBA, INICIO_VALIDACION, FIN_AJUSTE_VALIDACION = periodos().values()


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

    def __init__(self, inicio: str = P3_INICIO_ENTRENAMIENTO, fin: str = P3_FIN_ENTRENAMIENTO):
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
        ultimo_dia = viajes[FECHA_INICIO_COL].max().normalize()
        if pd.Timestamp(fin) > ultimo_dia:
            raise ValueError(f"fin={fin} es posterior a los datos ({ultimo_dia.date()})")
        # Si el entrenamiento llega a FECHA_FIN (reentrenamiento con todo el histórico), la
        # prueba interna queda vacía; ese periodo posterior todavía no existe en el Parquet.
        p = periodos(fin)
        self.inicio, self.fin = inicio, fin
        self.train = perfil_estaciones(viajes, inicio, fin)
        self.test = perfil_estaciones(viajes, p["inicio_prueba"], FECHA_FIN)
        self.ajuste_val = perfil_estaciones(viajes, inicio, p["fin_ajuste_validacion"])
        self.val = perfil_estaciones(viajes, p["inicio_validacion"], fin)
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
    resultado = {
        "train_silhouette_espacio_comun": silhouette_score(X_comun, grupos),
        "train_davies_bouldin_espacio_comun": davies_bouldin_score(X_comun, grupos),
        "train_calinski_harabasz_espacio_comun": calinski_harabasz_score(X_comun, grupos),
        "train_tamano_min_grupo": int(tamanos.min()),
        "train_prop_min_grupo": float(tamanos.min() / len(grupos)),
        "val_ari_estabilidad": ari_estabilidad(pipe_val, datos.ajuste_val, datos.val, datos.cols),
    }
    if len(datos.test):
        resultado["test_ari_estabilidad"] = ari_estabilidad(
            pipe, datos.train, datos.test, datos.cols
        )
    return resultado


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
            info_estaciones(leer_nomenclatura(NOMENCLATURA_CSV)), on="estacion_id"
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



def _registrar_modelo(pipe: Pipeline, X: pd.DataFrame) -> None:
    """Pipeline con firma y ejemplo de entrada; es lo que después se registra en el Registry."""
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
             artefactos: bool = True, modelo: bool | None = None) -> tuple[str, dict]:
    """Ajusta un modelo P3 y registra parámetros, métricas, etiquetas y artefactos en MLflow.

    Los parámetros se registran con los mismos nombres que `mlflow run -P`, salvo la semilla
    (`random_state`) y la fecha (`inicio_entrenamiento`), que pueden venir vacías desde el
    MLproject y resolverse en config.py. `artefactos` guarda gráficas, tablas y el modelo;
    `modelo` permite guardar solo el modelo (por omisión sigue a `artefactos`).
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
            "fin_entrenamiento": datos.fin,
            "validacion": f"{periodos(datos.fin)['inicio_validacion']}..{datos.fin}",
            "prueba": f"{periodos(datos.fin)['inicio_prueba']}..{FECHA_FIN}"
            if len(datos.test) else "sin prueba interna",
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
        if artefactos if modelo is None else modelo:
            _registrar_modelo(pipe, datos.train[datos.cols])
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
            "inicio_entrenamiento": origen.data.params.get("inicio_entrenamiento", ""),
            "fin_entrenamiento": origen.data.params.get("fin_entrenamiento", ""),
        },
    )
    p = origen.data.params
    metricas_run = origen.data.metrics
    resumen = ", ".join(
        f"{m}={metricas_run[m]:.3f}"
        for m in ("val_ari_estabilidad", "test_ari_estabilidad", "train_silhouette_espacio_comun")
        if m in metricas_run
    )
    cliente.update_model_version(
        NOMBRE_REGISTRO,
        version.version,
        description=(
            f"{origen.info.run_name} "
            f"({p.get('inicio_entrenamiento')}..{p.get('fin_entrenamiento')}): {resumen}"
        ),
    )
    cliente.set_registered_model_alias(NOMBRE_REGISTRO, alias, version.version)
    return version.version, mlflow.pyfunc.load_model(f"models:/{NOMBRE_REGISTRO}@{alias}")


def reentrenar(datos: DatosP3, alias: str = "challenger", artefactos: bool = True,
               **cambios) -> tuple[str, str]:
    """Versión alternativa: la receta de champion reentrenada con la ventana de `datos` (y los
    hiperparámetros de `cambios`, si se indican), registrada como nueva versión del mismo modelo
    con `alias`. No mueve champion: la promoción se decide después con evidencia (A/B testing).

    `artefactos=False` omite gráficas y tablas (no el modelo). Devuelve la versión nueva y la de
    champion.
    """
    if alias == "champion":
        raise ValueError("La versión alternativa no puede registrarse directamente como champion")
    cliente = mlflow.MlflowClient()
    try:
        base = cliente.get_model_version_by_alias(NOMBRE_REGISTRO, "champion")
    except mlflow.exceptions.MlflowException as error:
        raise ValueError(
            f"{NOMBRE_REGISTRO} no tiene versión champion; regístrala primero"
        ) from error
    params_base = cliente.get_run(base.run_id).data.params
    receta = {
        "model": params_base["model"], "k": int(params_base["k"]), "pca": int(params_base["pca"]),
        "cov": params_base["cov"], "linkage": params_base["linkage"],
        "n_init": int(params_base["n_init"]), "seed": int(params_base["random_state"]),
    }
    hp = {**receta, **cambios}
    ventana_base = (params_base["inicio_entrenamiento"], params_base["fin_entrenamiento"])
    if (datos.inicio, datos.fin) == ventana_base and hp == receta:
        raise ValueError(f"Misma ventana ({datos.inicio}..{datos.fin}) e hiperparámetros que "
                         f"champion (v{base.version}); cambia las fechas o los hiperparámetros")

    nombre = f"p3-{hp['model']}-k{hp['k']}-reentrenado-{datos.inicio}-a-{datos.fin}"
    run_id, metricas_nuevas = entrenar(datos, stage="retrain", run_name=nombre,
                                       artefactos=artefactos, modelo=True, **hp)
    version, _ = registrar(run_id, alias)
    cliente.set_model_version_tag(NOMBRE_REGISTRO, version, "version_base", base.version)

    # Comparación descriptiva con champion: métricas de entrenamiento de cada versión en su propia
    # ventana. Decidir cuál queda vigente corresponde a la evaluación sobre el periodo actual.
    metricas_base = cliente.get_run(base.run_id).data.metrics
    cliente.log_dict(run_id, {
        "champion": {"version": str(base.version), "ventana": list(ventana_base),
                     "hiperparametros": receta,
                     "metricas": {m: metricas_base.get(m) for m in metricas_nuevas}},
        alias: {"version": str(version), "ventana": [datos.inicio, datos.fin],
                "hiperparametros": hp, "metricas": metricas_nuevas},
    }, "tables/comparacion_con_champion.json")
    return str(version), str(base.version)


class ClusteringModel:
    """Compatibilidad con `python -m mi_bici_experiment.modeling.train`: baseline y candidato."""

    def ejecutar(self):
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        datos = DatosP3()
        entrenar(datos, "kmeans", P3_K_BASELINE, stage="baseline")
        entrenar(datos, "kmeans", P3_K_CANDIDATO, P3_PCA_COMPONENTES)
