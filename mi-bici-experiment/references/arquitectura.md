# Arquitectura de TC5061-Proyecto-Equipo12


Flujo implementado para ingesta, preparación de datos, entrenamiento y seguimiento de los tres problemas de MiBici. Las flechas continuas indican flujo de datos o resultados; las discontinuas, configuración y soporte.

```mermaid
flowchart TB
    fuente["MiBici · datos abiertos<br/>CSV mensuales de viajes y nomenclatura"]

    subgraph proyecto["Proyecto Python · mi-bici-experiment"]
        config["config.py + variables de entorno<br/>Rutas, fechas, parámetros y experimento MLflow"]
        dataset["dataset.py<br/>Descarga, lectura de CSV y conversión de fechas"]
        raw[("data/raw/<br/>CSV, trips.parquet y nomenclatura.parquet")]
        features["features.py<br/>Cruce de viajes con estación de origen"]
        interim[("data/interim/features.parquet<br/>Fechas, estaciones, coordenadas y capacidad")]

        subgraph entrenamiento["modeling/train.py · CLI Typer · comando todos"]
            p1["P1 · RegresionP1<br/>Viajes diarios, calendario y rezagos<br/>Regresión lineal · MAE y RMSE"]
            p2["P2 · RegresionLinealMultinomialP2<br/>Balance diario, rezagos y variables geoespaciales<br/>Regresión logística · precisión, recall y F1"]
            p3["P3 · ClusteringP3<br/>Perfiles horarios y resumen de uso<br/>K-means / escalado + PCA + K-means<br/>Silueta, Davies-Bouldin y estabilidad ARI"]
        end

        tracking["tracking.py<br/>Huella SHA-256 de CSV y metadatos Git"]
        plots["plots.py y gráficas de P1<br/>Balance, matrices de confusión,<br/>predicciones, mapas y perfiles"]
        reports[("reports/figures/<br/>Gráficas y tablas por experimento")]
        notebooks["notebooks/<br/>Exploración y análisis"]
        tests["tests/<br/>Pruebas de configuración y variables P3"]
    end

    subgraph seguimiento["Seguimiento de experimentos"]
        mlflow["MLflow Tracking<br/>Experimento: mibici-equipo12<br/>Parámetros, métricas, tags y artefactos"]
        registry["MLflow Model Registry<br/>Pipeline de clasificación P2"]
        modelos["Modelos como artefactos MLflow<br/>P2 y candidato P3"]
    end

    fuente --> dataset --> raw --> features --> interim
    interim --> p1
    interim --> p2
    interim --> p3
    raw -->|Nomenclatura| p3
    interim --> notebooks
    p1 --> plots
    p2 --> plots
    p3 --> plots
    plots --> reports
    p1 --> mlflow
    p2 --> mlflow
    p3 --> mlflow
    reports -->|Carga de artefactos| mlflow
    p2 --> registry
    p2 --> modelos
    p3 --> modelos
    config -.-> dataset
    config -.-> features
    config -.-> entrenamiento
    raw -.->|CSV para hash| tracking
    tracking -.-> p2
    tracking -.-> p3
    tests -.-> features

    classDef datos fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef modelo fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef seguimiento fill:#f3e8ff,stroke:#9333ea,color:#581c87
    class raw,interim,reports datos
    class p1,p2,p3 modelo
    class mlflow,registry,modelos seguimiento
```

## Ejecución

Desde `mi-bici-experiment/`:

```bash
python -m mi_bici_experiment.dataset
python -m mi_bici_experiment.features
python -m mi_bici_experiment.modeling.train todos
```

`todos` ejecuta P1, P2 y P3 en secuencia. Las variables específicas de cada problema se construyen en memoria mediante funciones de `features.py` durante el entrenamiento.

## Alcance actual

- MLflow usa la configuración del entorno; el diagrama no presupone un servidor ni un proveedor de almacenamiento. Los modelos P2 y P3 candidato se guardan como artefactos; P2 además se registra en el Model Registry. P1 registra métricas y gráficas.
- P2 y P3 usan `version_datos()`; P3 incorpora además los metadatos de `git_info()`. Los CSV y los almacenes locales de MLflow se excluyen de Git.
- `.github/workflows/mlflow.yml` contiene un flujo de GitHub Actions todavía incompleto: no automatiza el entrenamiento de extremo a extremo.
- `modeling/predict.py` es una plantilla. No hay un servicio de inferencia ni un despliegue implementado en el repositorio.
