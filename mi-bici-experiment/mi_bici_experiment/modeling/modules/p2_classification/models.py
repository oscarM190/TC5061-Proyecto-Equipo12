"""Entrenamiento y seguimiento de regresión logística y Random Forest para P2."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd
import typer
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici_experiment.config import (
    EQUIPO,
    FECHA_FIN,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    P2_COMBINACIONES,
    P2_DATASET_PARQUET,
    P2_FECHA_INICIO,
    P2_NOMBRE_REGISTRO,
    RANDOM_STATE,
    TRIPS_PARQUET,
)
from mi_bici_experiment.features import (
    agregar_balance_neto_y_normalizado,
    clasificacion_variable_objetivo_y_caracterizticas,
    creacion_variables_memoria_historica,
)
from mi_bici_experiment.plots import matriz_confucion_regresion_multiclass_clasificacion
from mi_bici_experiment.tracking import git_info, version_datos


class ClassificationModel:
    """Entrena, evalúa y registra el clasificador multinomial en MLflow."""

    def __init__(self, dataset_ml: pd.DataFrame):
        self.dataset_ml = dataset_ml
        self.numeric_features = [
            "dpcapacity",
            "Dia_Semana",
            "Es_Fin_Semana",
            "Mes",
            "Dia_Semana_Objetivo",
            "Es_Fin_Semana_Objetivo",
            "Mes_Objetivo",
            "Llegadas",
            "Salidas",
            "Llegadas_Por_Capacidad",
            "Salidas_Por_Capacidad",
            "Balance_Neto",
            "Balance_Normalizado",
            "Balance_Lag_1",
            "Balance_Lag_2",
            "Balance_Lag_3",
            "Balance_Lag_7",
            "Media_Movil_3d_Balance",
            "Media_Movil_7d_Balance",
            "Llegadas_Lag_1",
            "Llegadas_Lag_2",
            "Llegadas_Lag_3",
            "Llegadas_Lag_7",
            "Salidas_Lag_1",
            "Salidas_Lag_2",
            "Salidas_Lag_3",
            "Salidas_Lag_7",
            "Media_Movil_7d_Llegadas",
            "Media_Movil_7d_Salidas",
            "Media_Movil_7d_Llegadas_Por_Capacidad",
            "Media_Movil_7d_Salidas_Por_Capacidad",
        ]
        self.numeric_features += [
            f"{flow}_Por_Capacidad_Lag_{lag}"
            for flow in ("Llegadas", "Salidas")
            for lag in (1, 2, 3, 7)
        ]
        self.categorical_features = ["Estacion_Id"]
        self.features = self.numeric_features + self.categorical_features

        # Construcción del Pipeline de Machine Learning
        # El pipeline garantiza que la estandarización se ajuste solo con los datos de entrenamiento
        self.pipeline_lr = Pipeline(
            [
                (
                    "preprocesamiento",
                    ColumnTransformer(
                        [
                            ("numericas", StandardScaler(), self.numeric_features),
                        ]
                    ),
                ),
                (
                    "clasificador",
                    LogisticRegression(
                        max_iter=1000,  # Incrementado para asegurar convergencia con múltiples variables
                        class_weight="balanced",  # Crucial si la clase 1 (balanceado) domina la distribución
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        # Clone preprocessing so training either model cannot change the other's state.
        self.pipeline_rf = Pipeline(
            [
                ("preprocesamiento", clone(self.pipeline_lr.named_steps["preprocesamiento"])),
                (
                    "clasificador",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=16,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        bootstrap=True,
                        max_samples=0.8,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.rf_default_params = self.pipeline_rf.named_steps["clasificador"].get_params()

    def separar_datos(self):
        """Divide fechas únicas 70/15/15 y excluye objetivos que cruzan los cortes."""
        if "Fecha_Objetivo" not in self.dataset_ml:
            raise ValueError("Falta Fecha_Objetivo; regenera el dataset de P2 antes de entrenar.")
        dataset = self.dataset_ml.copy()
        for column in ("Fecha", "Fecha_Objetivo"):
            dataset[column] = pd.to_datetime(dataset[column], errors="raise").dt.normalize()
        if dataset[["Fecha", "Fecha_Objetivo"]].isna().any().any():
            raise ValueError("Las fechas de entrada y objetivo no pueden contener nulos.")
        if (dataset["Fecha_Objetivo"] <= dataset["Fecha"]).any():
            raise ValueError("La fecha objetivo debe ser posterior a la fecha de entrada.")
        dataset = dataset.sort_values(["Fecha", "Estacion_Id"]).reset_index(drop=True)
        fechas = dataset["Fecha"].drop_duplicates().tolist()
        corte_val = int(len(fechas) * 0.70)
        corte_test = int(len(fechas) * 0.85)
        if not 0 < corte_val < corte_test < len(fechas):
            raise ValueError(
                "No hay suficientes fechas únicas para separar train/validación/test."
            )
        inicio_val, inicio_test = fechas[corte_val], fechas[corte_test]
        train_mask = dataset["Fecha"] < inicio_val
        val_mask = (dataset["Fecha"] >= inicio_val) & (dataset["Fecha"] < inicio_test)
        purge_train = train_mask & (dataset["Fecha_Objetivo"] >= inicio_val)
        purge_val = val_mask & (dataset["Fecha_Objetivo"] >= inicio_test)
        train = dataset.loc[train_mask & ~purge_train].copy()
        val = dataset.loc[val_mask & ~purge_val].copy()
        test = dataset.loc[dataset["Fecha"] >= inicio_test].copy()
        if any(part.empty for part in (train, val, test)):
            raise ValueError(
                "Algún conjunto quedó vacío después de excluir objetivos entre periodos."
            )
        self.split_params = {
            "particion_unidad": "fechas_unicas",
            "inicio_validacion": str(inicio_val.date()),
            "inicio_test": str(inicio_test.date()),
            "n_fechas_total": len(fechas),
            "n_purgadas_train": int(purge_train.sum()),
            "n_purgadas_validacion": int(purge_val.sum()),
            "n_fechas_train": train["Fecha"].nunique(),
            "n_fechas_validacion": val["Fecha"].nunique(),
            "n_fechas_test": test["Fecha"].nunique(),
        }
        return train, val, test

    def entrenar(self):
        """Mantiene la entrada original para entrenar la regresión logística."""
        return self.entrenar_regresion_logistica()

    def entrenar_regresion_logistica(self):
        """Entrena y registra la regresión logística con sus parámetros actuales."""
        return self._entrenar_pipeline(
            self.pipeline_lr,
            "regresion-lineal-multinomial-clasificacion",
            "Regresión Logística Multinomial",
            "matriz_confusion_multinomial",
        )

    def entrenar_random_forest(self, hiperparametros=None, nombre=None, nested=False):
        """Entrena y registra Random Forest con la misma evaluación que la regresión."""
        pipeline = clone(self.pipeline_rf)
        params = {**self.rf_default_params, **(hiperparametros or {})}
        pipeline.set_params(clasificador=RandomForestClassifier(**params))
        self.pipeline_rf = pipeline
        return self._entrenar_pipeline(
            pipeline,
            "random-forest-clasificacion",
            "Random Forest",
            "matriz_confusion_random_forest" + (f"_{nombre}" if nombre else ""),
            run_name=f"P2-random-forest-{nombre}" if nombre else None,
            nested=nested,
        )

    def _entrenar_pipeline(
        self, pipeline, tipo, nombre_modelo, nombre_figura, run_name=None, nested=False
    ):
        """Entrena y registra un pipeline con la evaluación común de P2."""
        train, val, test = self.separar_datos()
        X_train, y_train = train[self.features], train["Riesgo_T1"]
        X_val, y_val = val[self.features], val["Riesgo_T1"]
        X_test, y_test = test[self.features], test["Riesgo_T1"]

        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        descripcion = (
            f"P2: {nombre_modelo} con escalado y estación one-hot; separación cronológica "
            "70/15/15 por fechas únicas y exclusión de objetivos que cruzan periodos. "
            "Predicción al cierre de T con flujos, capacidad, estación one-hot y calendario T+1. "
            "Métricas globales macro."
        )
        self.data_version = version_datos()
        with mlflow.start_run(
            run_name=run_name or tipo,
            nested=nested,
            tags={
                "equipo": EQUIPO,
                "pregunta": "P2",
                "tipo_modelo": tipo,
                **git_info(),
                "entorno": os.getenv("MIBICI_ENTORNO", "local"),
                "mlflow.note.content": descripcion,
            },
        ) as run:
            self.last_run_id = run.info.run_id
            self.metricas = {}
            params = {
                key: value
                for key, value in pipeline.get_params(deep=True).items()
                if key.startswith(
                    (
                        "preprocesamiento__numericas__",
                        "preprocesamiento__estacion__",
                        "clasificador__",
                    )
                )
            }
            params.update(
                {
                    "train_fraccion": 0.70,
                    "validacion_fraccion": 0.15,
                    "test_fraccion": 0.15,
                    "shuffle": False,
                    "version_datos": self.data_version,
                    "features": json.dumps(self.features),
                    "categorical_features": json.dumps(self.categorical_features),
                    "momento_prediccion": "cierre_dia_T",
                    "codificacion_estacion": "one_hot_handle_unknown_ignore",
                    "objetivo": "Riesgo_T1",
                    "promedio_metricas": "macro",
                    "n_train": len(X_train),
                    "n_validacion": len(X_val),
                    "n_test": len(X_test),
                    "fecha_inicio": P2_FECHA_INICIO,
                    "fecha_fin": FECHA_FIN,
                    "random_state": pipeline.named_steps["clasificador"].random_state,
                }
            )
            params.update(self.split_params)
            mlflow.log_params(params)
            # Entrenamiento del Modelo
            pipeline.fit(X_train, y_train)

            # Evaluación sin volver a ajustar el modelo ni el escalador.
            for name, X_eval, y_eval in (
                ("Validación", X_val, y_val),
                ("Prueba", X_test, y_test),
            ):
                y_pred = pipeline.predict(X_eval)
                print(f"Reporte de Clasificación - {name} ({nombre_modelo}):")
                print(
                    classification_report(
                        y_eval,
                        y_pred,
                        labels=[0, 1, 2],
                        target_names=["Sin Bicis (0)", "Balanceado (1)", "Sin Anclajes (2)"],
                        zero_division=0,
                    )
                )
                report = classification_report(
                    y_eval, y_pred, labels=[0, 1, 2], output_dict=True, zero_division=0
                )
                prefix = "val" if name == "Validación" else "test"
                metrics = {
                    f"{prefix}_{metric}": report["macro avg"][metric]
                    for metric in ("precision", "recall", "f1-score")
                }
                self.metricas.update(metrics)
                mlflow.log_metrics(metrics)
                for label in ("0", "1", "2"):
                    mlflow.log_metrics(
                        {
                            f"{prefix}_clase_{label}_{metric}": report[label][metric]
                            for metric in ("precision", "recall", "f1-score")
                        }
                    )
                matrix_path = matriz_confucion_regresion_multiclass_clasificacion(
                    y_eval, y_pred, conjunto=name, modelo=nombre_modelo, nombre_base=nombre_figura
                )
                mlflow.log_artifact(str(matrix_path), artifact_path="figures")
            mlflow.log_artifact(
                str(FIGURES_DIR / "balance_normalizado.png"), artifact_path="figures"
            )
            # Log floating-point inputs so nullable integer features remain schema-compatible.
            input_example = X_train.head(5).astype("float64")
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name=tipo,
                registered_model_name=tipo if not nested else None,
                signature=infer_signature(input_example, pipeline.predict(input_example)),
                input_example=input_example,
                # These trees were fitted here; explicitly trust only their storage type.
                **(
                    {"skops_trusted_types": ["sklearn.tree._tree.Tree"]}
                    if isinstance(pipeline.named_steps["clasificador"], RandomForestClassifier)
                    else {}
                ),
            )
            self.last_model_uri = model_info.model_uri
        return pipeline

    def registrar_mejor_modelo(self, mejor):
        """Registra el artefacto ganador y asigna champion con el linaje de su run."""
        client = mlflow.MlflowClient()
        winner_run = client.get_run(mejor["run_id"])
        version = mlflow.register_model(model_uri=mejor["model_uri"], name=P2_NOMBRE_REGISTRO)
        client.update_registered_model(
            P2_NOMBRE_REGISTRO,
            description="P2: riesgo de desequilibrio diario por estación mediante Random Forest.",
        )
        client.update_model_version(
            P2_NOMBRE_REGISTRO,
            version.version,
            description=(
                f"Mejor candidato por F1 macro de validación: {mejor['val_f1_score']:.6f}; "
                f"combinación {mejor['combinacion']}; run {mejor['run_id']}."
            ),
        )
        tags = {
            key: value
            for key, value in winner_run.data.tags.items()
            if key in ("equipo", "pregunta", "git_commit", "git_branch", "git_dirty", "entorno")
        }
        tags.update(
            {
                "data_version": winner_run.data.params["version_datos"],
                "source_run_id": mejor["run_id"],
                "combinacion": mejor["combinacion"],
                "criterio": "val_f1-score",
                "val_f1_score": str(mejor["val_f1_score"]),
            }
        )
        for key, value in tags.items():
            client.set_model_version_tag(P2_NOMBRE_REGISTRO, version.version, key, value)
        client.set_registered_model_alias(P2_NOMBRE_REGISTRO, "champion", version.version)
        self.mejor_version = version.version
        print(f"{P2_NOMBRE_REGISTRO} v{version.version} registrado con alias champion")
        return version

    def buscar_hiperparametros(self, combinaciones=None):
        """Compara configuraciones en runs hijos y selecciona por F1 macro de validación."""
        combinaciones = P2_COMBINACIONES if combinaciones is None else combinaciones
        if not combinaciones:
            raise ValueError("No hay configuraciones de Random Forest para entrenar.")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        resultados = []
        with mlflow.start_run(
            run_name="P2-random-forest-search",
            tags={
                "equipo": EQUIPO,
                "pregunta": "P2",
                "tipo_modelo": "random-forest-search",
            },
        ):
            for nombre, params in combinaciones.items():
                self.entrenar_random_forest(params, nombre=nombre, nested=True)
                resultados.append(
                    {
                        "combinacion": nombre,
                        "run_id": self.last_run_id,
                        "model_uri": self.last_model_uri,
                        "val_f1_score": self.metricas["val_f1-score"],
                        "parametros": self.pipeline_rf.named_steps["clasificador"].get_params(),
                    }
                )
            mejor = max(resultados, key=lambda result: result["val_f1_score"])
            mlflow.log_params(
                {"mejor_combinacion": mejor["combinacion"], "criterio": "val_f1-score"}
            )
            mlflow.log_metric("best_val_f1_score", mejor["val_f1_score"])
            mlflow.set_tag("best_run_id", mejor["run_id"])
            mlflow.log_dict(resultados, "busqueda/resultados.json")
            version = self.registrar_mejor_modelo(mejor)
            mlflow.log_params(
                {
                    "modelo_registrado": P2_NOMBRE_REGISTRO,
                    "version_registrada": version.version,
                    "alias_modelo": "champion",
                }
            )
        self.resultados_busqueda = resultados
        self.mejor_resultado = mejor
        return mejor

    @staticmethod
    def ejecutar(hiperparametros=None, solo_baseline=False):
        df = pd.read_parquet(
            TRIPS_PARQUET,
            columns=[
                "inicio_del_viaje",
                "fin_del_viaje",
                "origen_id",
                "destino_id",
                "dpcapacity",
            ],
        ).rename(
            columns={
                "inicio_del_viaje": "Inicio_del_viaje",
                "fin_del_viaje": "Fin_del_viaje",
                "origen_id": "Origen_Id",
                "destino_id": "Destino_Id",
            }
        )
        balance_diario = agregar_balance_neto_y_normalizado(df)
        dataset_ml = clasificacion_variable_objetivo_y_caracterizticas(balance_diario)
        dataset_ml = creacion_variables_memoria_historica(dataset_ml)
        # Keep earlier rows as history for lag calculations, but train only from this date onward.
        dataset_ml = dataset_ml.loc[dataset_ml["Fecha"] >= pd.Timestamp(P2_FECHA_INICIO)].copy()

        # Persist the exact date-filtered dataset used by the P2 model.
        P2_DATASET_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        dataset_ml.to_parquet(P2_DATASET_PARQUET, index=False)
        dataset_ml = pd.read_parquet(P2_DATASET_PARQUET)
        modelo = ClassificationModel(dataset_ml)
        if solo_baseline:
            return modelo.entrenar_regresion_logistica()
        if hiperparametros is None:
            modelo.entrenar_regresion_logistica()
            return modelo.buscar_hiperparametros()
        return modelo.entrenar_random_forest(hiperparametros, nombre="personalizada")


app = typer.Typer()


@app.command()
def main(
    n_estimators: int | None = None,
    max_depth: int | None = None,
    min_samples_split: int | None = None,
    min_samples_leaf: int | None = None,
    max_features: str | None = None,
    bootstrap: bool | None = None,
    max_samples: float | None = None,
    class_weight: str | None = None,
    random_state: int | None = None,
    n_jobs: int | None = None,
):
    """Sin opciones ejecuta P2_COMBINACIONES; con opciones entrena un solo Random Forest."""
    elegidos = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "max_samples": max_samples,
        "class_weight": class_weight,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    elegidos = {key: value for key, value in elegidos.items() if value is not None}
    if not elegidos:
        return ClassificationModel.ejecutar()
    # Same convention as P1: unspecified options inherit modelo_3 from config.py.
    params = {**P2_COMBINACIONES["modelo_3"], **elegidos}
    if max_features is not None and max_features not in ("sqrt", "log2"):
        try:
            params["max_features"] = float(max_features)
        except ValueError:
            raise typer.BadParameter(
                "Usa sqrt, log2 o una fracción entre 0 y 1 para max-features."
            )
    if class_weight == "none":
        params["class_weight"] = None
    if not params.get("bootstrap", True):
        if max_samples is not None:
            raise typer.BadParameter("max-samples requiere bootstrap.")
        params["max_samples"] = None
    try:
        RandomForestClassifier(**params)._validate_params()
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    return ClassificationModel.ejecutar(params)


if __name__ == "__main__":
    app()
