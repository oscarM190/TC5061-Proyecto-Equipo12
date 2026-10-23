"""P2 multinomial logistic regression training implementation."""

import json
import os

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici_experiment.config import (
    EQUIPO,
    FECHA_FIN,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    P2_DATASET_PARQUET,
    P2_FECHA_INICIO,
    P2_FEATURES_PARQUET,
    RANDOM_STATE,
)
from mi_bici_experiment.features import (
    agregar_balance_neto_y_normalizado,
    clasificacion_variable_objetivo_y_caracterizticas,
    creacion_variables_memoria_historica,
    main as generar_features,
)
from mi_bici_experiment.plots import matriz_confucion_regresion_multiclass_clasificacion
from mi_bici_experiment.tracking import git_info, version_datos


class ClassificationModel:
    """Entrena, evalúa y registra el clasificador multinomial en MLflow."""

    def __init__(self, dataset_ml: pd.DataFrame):
        self.dataset_ml = dataset_ml
        self.features = [
            "Dia_Semana",
            "Es_Fin_Semana",
            "Mes",
            "Balance_Lag_7",
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
        ]

        # Construcción del Pipeline de Machine Learning
        # El pipeline garantiza que la estandarización se ajuste solo con los datos de entrenamiento
        self.pipeline_lr = Pipeline(
            [
                ("scaler", StandardScaler()),
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

    def entrenar(self):
        """Separa los datos 70/15/15 y registra el entrenamiento y sus resultados."""
        X = self.dataset_ml[self.features]
        y = self.dataset_ml["Riesgo_T1"]

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
        with mlflow.start_run(
            run_name=tipo,
            tags={
                "equipo": EQUIPO,
                "pregunta": "P2",
                "tipo_modelo": tipo,
                **git_info(),
                "entorno": os.getenv("MIBICI_ENTORNO", "local"),
                "mlflow.note.content": descripcion,
            },
        ):
            params = {
                key: value
                for key, value in self.pipeline_lr.get_params(deep=True).items()
                if key.startswith(("scaler__", "clasificador__"))
            }
            params.update(
                {
                    "train_fraccion": 0.70,
                    "validacion_fraccion": 0.15,
                    "test_fraccion": 0.15,
                    "shuffle": False,
                    "version_datos": self.data_version,
                    "features": json.dumps(self.features),
                    "objetivo": "Riesgo_T1",
                    "promedio_metricas": "macro",
                    "n_train": len(X_train),
                    "n_validacion": len(X_val),
                    "n_test": len(X_test),
                    "fecha_inicio": P2_FECHA_INICIO,
                    "fecha_fin": FECHA_FIN,
                    "random_state": RANDOM_STATE,
                }
            )
            mlflow.log_params(params)
            # Entrenamiento del Modelo
            self.pipeline_lr.fit(X_train, y_train)

            # Evaluación sin volver a ajustar el modelo ni el escalador.
            for name, X_eval, y_eval in (
                ("Validación", X_val, y_val),
                ("Prueba", X_test, y_test),
            ):
                y_pred = self.pipeline_lr.predict(X_eval)
                print(f"Reporte de Clasificación - {name} (Baseline Multinomial):")
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
                mlflow.log_metrics(
                    {
                        f"{prefix}_{metric}": report["macro avg"][metric]
                        for metric in ("precision", "recall", "f1-score")
                    }
                )
                for label in ("0", "1", "2"):
                    mlflow.log_metrics(
                        {
                            f"{prefix}_clase_{label}_{metric}": report[label][metric]
                            for metric in ("precision", "recall", "f1-score")
                        }
                    )
                matrix_path = matriz_confucion_regresion_multiclass_clasificacion(
                    y_eval, y_pred, conjunto=name
                )
                mlflow.log_artifact(str(matrix_path), artifact_path="figures")
            mlflow.log_artifact(
                str(FIGURES_DIR / "balance_normalizado.png"), artifact_path="figures"
            )
            # Log floating-point inputs so nullable integer features remain schema-compatible.
            input_example = X_train.head(5).astype("float64")
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
        features_path = P2_FEATURES_PARQUET
        if not features_path.is_file():
            generar_features()
        df = pd.read_parquet(features_path)
        balance_diario = agregar_balance_neto_y_normalizado(df)
        dataset_ml = clasificacion_variable_objetivo_y_caracterizticas(balance_diario)
        dataset_ml = creacion_variables_memoria_historica(dataset_ml)
        # Keep earlier rows as history for lag calculations, but train only from this date onward.
        dataset_ml = dataset_ml.loc[
            dataset_ml["Fecha"] >= pd.Timestamp(P2_FECHA_INICIO)
        ].copy()

        # Persist the exact date-filtered dataset used by the P2 model.
        P2_DATASET_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        dataset_ml.to_parquet(P2_DATASET_PARQUET, index=False)
        dataset_ml = pd.read_parquet(P2_DATASET_PARQUET)

        modelo = ClassificationModel(dataset_ml)
        modelo.entrenar()
