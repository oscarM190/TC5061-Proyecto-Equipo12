# TC5061 Proyecto MiBici, Equipo 12

[![MiBici CI/CD](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/actions/workflows/mlflow.yml/badge.svg)](https://github.com/oscarM190/TC5061-Proyecto-Equipo12/actions/workflows/mlflow.yml)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador del curso TC5061 Operaciones de Aprendizaje Automático (MLOps), Tec de Monterrey.
El equipo propone y valida soluciones de Machine Learning para la operación de MiBici, el sistema de
bicicleta pública del Área Metropolitana de Guadalajara, con datos reproducibles y experimentos
registrados en MLflow.

## Preguntas del proyecto

| # | Pregunta | Tarea | Formulación | Usuario |
|---|----------|-------|-------------|---------|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Viajes iniciados por estación y día, con rezagos de 1 y 7 días y promedio móvil de 7 días | Operador del sistema |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Estado de cada estación al día siguiente (sin bicicletas, balanceada o sin anclajes) a partir del balance neto diario (llegadas − salidas) dividido entre su capacidad | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Grupos de estaciones según su perfil horario de salidas y llegadas, peso del fin de semana, uso diario y balance | Planeación urbana |

## Datos

- **Fuente:** [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). Viajes por
  mes (`datos_abiertos_AAAA_MM.csv`) y nomenclatura de estaciones (`nomenclatura_2026_09.csv`). Las URL
  exactas están en `MI_BICI_CSVS` de `mi_bici_experiment/config.py`.
- **Rango:** octubre de 2025 a septiembre de 2026 (12 meses; septiembre de 2026 es el último mes
  publicado). Cubre un año completo de estacionalidad.
- **Tamaño:** 4,497,877 viajes crudos en 12 archivos. La limpieza de P3 (sin duraciones no positivas o
  mayores a 6 h ni devoluciones en la misma estación en menos de 60 s) deja 4,354,979. La nomenclatura
  tiene 484 estaciones (14 retiradas).
- **Variables usadas:** `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id` de los viajes, y
  `id`, `name`, `latitude`, `longitude`, `dpcapacity` de la nomenclatura.
- **Condiciones de uso:** el portal publica los datos para explorarlos y reutilizarlos, sin una licencia
  explícita. El equipo los usa con fines académicos, cita la fuente y no los redistribuye: `data/` está
  excluida de Git.
- **Datos personales:** los viajes incluyen `Usuario_Id`, `Genero` y `Año_de_nacimiento`. Ningún modelo
  los usa: `features.parquet` solo conserva las columnas de la lista anterior, y no se registran en MLflow.
  `data/raw/trips.parquet` todavía los conserva (no se versiona); eliminarlos en la ingesta está pendiente.
- **Partición temporal** (sin mezclar fechas):

| Pregunta | Entrenamiento | Evaluación |
|----------|---------------|------------|
| P1 | Primer 80% de las fechas | Último 20% de las fechas |
| P3 | 2025-10 a 2026-06 | 2026-07 a 2026-09, solo para medir la estabilidad de los grupos |

Las variables de P1 usan `shift` por estación y las de P3 se calculan solo con el periodo de
entrenamiento, así que ninguna usa información posterior al momento de la predicción.

## Estructura

El proyecto se generó con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
dentro de `mi-bici-experiment/`:

```
├── .github/workflows/mlflow.yml   <- Pipeline de GitHub Actions
├── README.md                      <- Este documento
└── mi-bici-experiment
    ├── Makefile                   <- Comandos de conveniencia (make requirements, make test)
    ├── pyproject.toml             <- Metadatos del paquete; Python 3.14.7
    ├── requirements.txt           <- Dependencias con versión fija
    ├── setup.cfg                  <- Configuración de flake8
    ├── data                       <- Datos de MiBici (no se versionan)
    │   ├── raw                    <- CSV originales, trips.parquet y nomenclatura.parquet
    │   └── interim                <- features.parquet, entrada de los modelos
    ├── docs                       <- Documentación mkdocs de la plantilla
    ├── models                     <- Modelos serializados (no se versionan; se registran en MLflow)
    ├── notebooks                  <- Exploración, con nombre 1.0-iniciales-descripcion.ipynb
    ├── references                 <- ML Canvas y diagrama de arquitectura
    ├── reports/figures            <- Figuras generadas
    ├── tests                      <- Pruebas con pytest
    └── mi_bici_experiment         <- Código del proyecto
        ├── config.py              <- Rutas, fechas, semilla, columnas y parámetros (único lugar)
        ├── dataset.py             <- Descarga de CSV y conversión a Parquet
        ├── features.py            <- Variables de P1 y P3
        ├── plots.py               <- Gráficas que se registran como artefactos
        └── modeling
            └── train.py           <- Entrenamiento y registro en MLflow (P1 y P3)
```

## Instalación

Requiere Python 3.14.7 y Git. Desde un clon limpio:

```bash
git clone https://github.com/oscarM190/TC5061-Proyecto-Equipo12.git
cd TC5061-Proyecto-Equipo12/mi-bici-experiment
python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Todos los comandos siguientes se ejecutan desde `mi-bici-experiment/`.

## Ingesta de datos

```bash
python -m mi_bici_experiment.dataset     # descarga los CSV y genera data/raw/*.parquet
python -m mi_bici_experiment.features    # genera data/interim/features.parquet
```

La descarga omite los archivos que ya existen, reintenta hasta tres veces y deja intactos los CSV en
`data/raw/`.

## Experimento base

```bash
python -m mi_bici_experiment.modeling.train regresion-p1    # P1
python -m mi_bici_experiment.modeling.train clustering-p3   # P3
python -m pytest tests                                      # pruebas
mlflow ui --backend-store-uri sqlite:///mlflow.db           # http://127.0.0.1:5000
```

Todos los runs se registran en el experimento `mibici-equipo12`:

| Pregunta | Run | Modelo | Métricas |
|----------|-----|--------|----------|
| P1 | `P1-regresion-lineal-baseline` | Regresión lineal con rezagos y calendario | `test_mae`, `test_rmse` |
| P2 | `regresion-lineal-multinomial-clasificacion` | Regresión logística multinomial con escalado (#5) | precisión, recall y F1 macro en validación y prueba |
| P3 | `P3-kmeans-baseline-k4` | K-means, k = 4, perfil horario de salidas | `train_silhouette_espacio_comun`, `train_davies_bouldin_espacio_comun`, `test_ari_estabilidad` |
| P3 | `P3-pca-kmeans-candidato` | Estandarización + PCA + k-means, k por silueta | igual |

En P3, `train_silhouette` y `train_davies_bouldin` se miden en el espacio de cada modelo y no se
comparan entre runs; las versiones `_espacio_comun` usan el perfil completo estandarizado.

## Estado del proyecto

| Componente | Estado | Pull Request |
|------------|--------|--------------|
| Estructura ccds | Integrado | `main` |
| Ingesta y Parquet | Integrado | #1 (`task/populate_data_scripts`) |
| P1 baseline | Integrado | #2 (`p1-regresion-modelo`) |
| P3 baseline y candidato | En revisión; sin conflictos con `main` | #3 (`feat/p3-clustering`) |
| P2 baseline | En revisión | #5 (`task/clasification-model`) |
| Script que entrena todos los modelos | En revisión; llega con #5 y P3 se agrega al integrar #3 | #5 |
| Versionado de datos | Huella SHA-256 de los CSV crudos en `data_version` (`tracking.py`, #3); manifiesto con conteo de registros pendiente | #3 |
| Pipeline de GitHub Actions | Pendiente; `mlflow.yml` no termina (`runs-on` y ruta de `requirements.txt`) | |
| Pruebas | `python -m pytest tests` pasa (7 pruebas) con #3 | #3 |

## Buenas prácticas aplicadas

| Práctica | Cómo se aplica |
|----------|----------------|
| Un experimento y runs con nombre descriptivo | Experimento `mibici-equipo12`; runs `P<n>-<modelo>` |
| Registrar lo necesario para reproducir | Fechas, partición, semilla (`RANDOM_STATE`), modelo e hiperparámetros como parámetros de cada run |
| Ligar el run con el código y la pregunta | Etiquetas `equipo`, `pregunta`, `tipo_modelo`, `git_commit`, `git_branch` y `git_dirty` (P3; pendiente en P1) |
| Métricas con el conjunto en el nombre | `test_mae`, `train_silhouette`, `test_ari_estabilidad`, etc. |
| Sin credenciales ni datos personales | Las columnas de usuario no llegan a `features.parquet`; `.env` se excluye de Git |
| Notebooks solo para exploración | `notebooks/1.0-<iniciales>-<descripcion>.ipynb`; la lógica vive en `mi_bici_experiment/` |
| Configuración en un solo lugar | Rutas, fechas, semilla y columnas en `config.py` |
| Dependencias fijas | `requirements.txt` con versiones fijas y Python 3.14.7 |
| Nada de datos, modelos ni MLflow en Git | `.gitignore` excluye `data/`, `mlruns/`, `mlartifacts/` y `mlflow.db` y modelos serializados (`*.pkl`, `*.joblib`, `*.skops`, `*.onnx`), estos dos últimos desde #3 |

## Flujo de trabajo con Git

Se sigue [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):

1. `main` está protegida: no admite push directo; todo cambio entra por Pull Request.
2. Cada cambio se hace en una rama creada desde `main`.
3. Se abre un Pull Request hacia `main`; otro integrante lo revisa y lo aprueba antes de integrarlo.
4. Tras la integración se borra la rama.

**Ramas:** `<tipo>/<descripcion-corta>`, con tipo `feat`, `fix`, `docs`, `test`, `ci`, `refactor`,
`chore` o `task`. Ejemplos: `feat/p3-clustering`, `task/populate_data_scripts`, `docs/readme`.

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/),
`<tipo>(<alcance>): <descripción>`. Ejemplos: `feat(p3): clustering de estaciones`,
`fix(dataset): eliminar columnas personales`.

**Pull Requests:** cada integrante abre al menos uno y revisa al menos uno de otro integrante. La
descripción indica qué cambia, cómo se probó y de qué otro PR depende.

## Integrantes

| Integrante | GitHub | Rol |
|------------|--------|-----|
| Oscar Eduardo Morales Portuondo | [@oscarM190](https://github.com/oscarM190) | Estructura ccds, ingesta de datos (`dataset.py`) y P2 (clasificación) |
| Julián Hazael Martínez Hernández | [@julianhmh01](https://github.com/julianhmh01) | P1 (regresión) |
| Eduardo Sebastián Vela Alanís | [@eduardoVela2022](https://github.com/eduardoVela2022) | EDA (`notebooks/eda_mibici_trips.ipynb`), ML Canvas, revisión de Pull Requests e integración a `main` |
| Jordan Piero Borda Colque | [@neronapier](https://github.com/neronapier) | P3 (clustering) y documentación (README) |

## Referencias

- Dorard, L. (2018, 9 de marzo). *From data to AI with the Machine Learning Canvas (Part I)*. Own Machine Learning.
- Huyen, C. (2022). *Designing machine learning systems*. O'Reilly Media.
- MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. https://www.mibici.net/es/datos-abiertos/
- MLflow. (s.f.). *MLflow Tracking*. https://mlflow.org/docs/latest/ml/tracking/
- Treveil, M., et al. (2020). *Introducing MLOps*. O'Reilly Media.
- Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.
