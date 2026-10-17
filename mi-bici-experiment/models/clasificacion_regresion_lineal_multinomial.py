import typer
import sys
import os
import json
from pathlib import Path
from mi_bici_experiment.config import INTERIM_DATA_DIR, FIGURES_DIR
from mi_bici_experiment.plots import matriz_confucion_regresion_multiclass_clasificacion
from mi_bici_experiment.features import agregar_balance_neto_y_normalizado, clasificacion_variable_objetivo_y_caracterizticas, creacion_variables_memoria_historica, creacion_variables_geoespaciales

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
app = typer.Typer()

import mlflow
import mlflow.sklearn
from scripts.git_helper import version_datos
from mlflow.models import infer_signature
from mi_bici_experiment.config import MLFLOW_EXPERIMENT

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

def regresion_lineal_multinomial(dataset_ml):
    features = [
        'dpcapacity', 'Dia_Semana', 'Es_Fin_Semana', 'Mes', 
        'Balance_Lag_1', 'Balance_Lag_2', 'Balance_Lag_7',
        'Estado_Lag_1', 'Estado_Lag_7', 
        'Media_Movil_3d', 'Media_Movil_7d', 'Volatilidad_3d',
        'Zona_Espacial', 'Densidad_500m'
    ]

    X = dataset_ml[features]
    y = dataset_ml['Riesgo_T1']

    # Separación cronológica: 70 % entrenamiento, 15 % validación y 15 % prueba.
    # El dataset debe estar ordenado por fecha antes de llamar a esta función.
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.5, shuffle=False
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Construcción del Pipeline de Machine Learning
    # El pipeline garantiza que la estandarización se ajuste solo con los datos de entrenamiento
    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clasificador', LogisticRegression(
            max_iter=1000, # Incrementado para asegurar convergencia con múltiples variables
            class_weight='balanced', # Crucial si la clase 1 (balanceado) domina la distribución
            random_state=42
        ))
    ])

    tipo = "regresion-lineal-multinomial-clasificacion"
    EQUIPO = os.getenv("EQUIPO", "equipo12")
    descripcion = (
        "P3: regresión logística multinomial con escalado; separación cronológica "
        "70 % entrenamiento, 15 % validación y 15 % prueba. Métricas globales macro."
    )
    data_version = version_datos()
    with mlflow.start_run(run_name=tipo, tags={
        "equipo": EQUIPO,
        "pregunta": "P3",
        "tipo_modelo": tipo,
        "entorno": os.getenv("MIBICI_ENTORNO", "local"),
        "mlflow.note.content": descripcion,
    }):
        params = {
            key: value for key, value in pipeline_lr.get_params(deep=True).items()
            if key.startswith(("scaler__", "clasificador__"))
        }
        params.update({
            "train_fraccion": 0.70, "validacion_fraccion": 0.15, "test_fraccion": 0.15,
            "shuffle": False, "version_datos": data_version,
            "features": json.dumps(features), "objetivo": "Riesgo_T1",
            "promedio_metricas": "macro", "n_train": len(X_train),
            "n_validacion": len(X_val), "n_test": len(X_test),
        })
        mlflow.log_params(params)
        # Entrenamiento del Modelo
        pipeline_lr.fit(X_train, y_train)

        # Evaluación sin volver a ajustar el modelo ni el escalador.
        for name, X_eval, y_eval in (
            ('Validación', X_val, y_val),
            ('Prueba', X_test, y_test),
        ):
            y_pred = pipeline_lr.predict(X_eval)
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
            prefix = "validacion" if name == "Validación" else "prueba"
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
            pipeline_lr,
            name=tipo,
            registered_model_name=tipo,
            signature=infer_signature(input_example, pipeline_lr.predict(input_example)),
            input_example=input_example,
        )

@app.command()
def main():
    df = pd.read_parquet(INTERIM_DATA_DIR / 'features.parquet')
    balance_diario = agregar_balance_neto_y_normalizado(df)
    dataset_ml = clasificacion_variable_objetivo_y_caracterizticas(balance_diario)
    dataset_ml = creacion_variables_memoria_historica(dataset_ml)
    dataset_ml = creacion_variables_geoespaciales(dataset_ml, df)
    regresion_lineal_multinomial(dataset_ml)

if __name__ == "__main__":
    app()