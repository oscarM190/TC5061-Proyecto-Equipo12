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
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Grupos de estaciones según el peso del fin de semana, la intensidad de uso diaria y el balance llegadas/salidas (la ingesta trunca las fechas al día) | Planeación urbana |

## Datos

- **Fuente:** [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). Viajes por
  mes (`datos_abiertos_AAAA_MM.csv`) y nomenclatura de estaciones (`nomenclatura_2026_09.csv`). Las URL
  exactas están en `MI_BICI_CSVS` de `mi_bici_experiment/config.py`.
- **Rango:** octubre de 2024 a septiembre de 2026 (24 meses; septiembre de 2026 es el último mes
  publicado). Cubre dos años completos de estacionalidad.
- **Tamaño:** 9,041,965  viajes crudos en 24 archivos, y un archivo de nomenclatura que contiene 484 estaciones.
- **Variables usadas:** `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id` de los viajes, y
  `id`, `name`, `latitude`, `longitude`, `dpcapacity` de la nomenclatura.
- **Condiciones de uso:** el portal publica los datos para explorarlos y reutilizarlos, sin una licencia
  explícita. El equipo los usa con fines académicos, cita la fuente y no los redistribuye: `data/` está
  excluida de Git.
- **Datos personales:** los viajes incluyen `Usuario_Id`, `Genero` y `Año_de_nacimiento`. Ningún modelo
  los usa: `features.parquet` solo conserva las columnas de la lista anterior, y no se registran en MLflow.
- **Partición temporal** 70% para entrenamiento, 15% para validación y 15% para pruebas.

## Estructura

El proyecto se generó con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
dentro de `mi-bici-experiment/`:

```
├── .github/workflows/mlflow.yml   <- Pipeline de GitHub Actions
├── MLproject                      <- Configuración del MLflow Project
├── python_env.yaml                <- Entorno de MLflow (Python y dependencias)
├── README.md                      <- Este documento
└── mi-bici-experiment
    ├── Makefile                   <- Comandos de conveniencia (make requirements, make test)
    ├── pyproject.toml             <- Metadatos del paquete; Python 3.14.7
    ├── requirements.txt           <- Dependencias con versión fija
    ├── setup.cfg                  <- Configuración de flake8
    ├── data                       <- Datos de MiBici
    │   ├── raw                    <- CSVs originales de viajes y nomenclatura
    │   ├── interim                <- Datos unificados de viajes y nomenclatura
    │   └── processed              <- Datos procesados para entrenar modelos
    ├── docs                       <- Documentación mkdocs de la plantilla
    ├── models                     <- Modelos serializados
    ├── notebooks                  <- Notebooks de Jupyter
    │   └── images                 <- Imágenes usadas en los notebooks
    ├── references                 <- ML Canvas y diagrama de arquitectura
    ├── reports/figures            <- Figuras generadas por los modelos
    ├── tests                      <- Pruebas con pytest
    └── mi_bici_experiment         <- Código del proyecto
        ├── config.py              <- Rutas, fechas, semilla, columnas y parámetros
        ├── dataset.py             <- Descarga de CSV y conversión a Parquet
        ├── features.py            <- Variables de los modelos
        ├── plots.py               <- Gráficas que se registran como artefactos
        ├── tracking.py            <- Registro de experimentos y artefactos en MLflow
        └── modeling
            ├── predict.py         <- Predicciones con modelos registrados
            ├── run.py             <- Ejecución de los flujos de modelado
            ├── train.py           <- Entrenamiento de modelos
            └── modules             <- Implementaciones de modelos para P1, P2 y P3
                ├── p1_regression
                │   └── models.py
                ├── p2_classification
                │   └── models.py
                └── p3_clustering
                    ├── estimadores.py
                    └── models.py
```

## Instalación

Requiere Python 3.14.7 y Git. Desde un clon limpio:

```bash
git clone https://github.com/oscarM190/TC5061-Proyecto-Equipo12.git
cd TC5061-Proyecto-Equipo12/mi-bici-experiment
python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # el paquete, para importarlo desde notebooks/ (o: make requirements)
```

Todos los comandos siguientes se ejecutan desde la raíz del repositorio.

## MLflow Project

El repositorio es un MLflow Project: [`MLproject`](MLproject) declara el entorno
([`python_env.yaml`](python_env.yaml), con las mismas versiones fijas de `requirements.txt`) y los puntos
de entrada. Todas las corridas de la actividad 5 se lanzan con `mlflow run` desde la raíz del repositorio.

### Servidor de tracking

El Model Registry necesita un almacenamiento con base de datos. En una terminal aparte, desde la raíz:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000      # Windows: set MLFLOW_TRACKING_URI=...
```

### Puntos de entrada

| Entry point | Parámetros | Qué hace |
|-------------|------------|----------|
| `ingest` | | Descarga los CSV, genera el Parquet y `data_manifest.json` |
| `train` | `question` (p1, p2, p3), `model` y sus hiperparámetros | Entrena y registra una corrida |
| `search` | `question`, `model`, `ks`, `pcas` | Búsqueda con nested runs: una corrida padre y una hija por combinación |
| `register` | `question`, `run_id`, `alias` | Registra el modelo de una corrida en el Model Registry y le asigna un alias |

Los hiperparámetros se pasan con `-P` aunque no estén declarados en el `MLproject`: MLflow los agrega al
comando (`-P k=3` llega como `--k 3`), así cada pregunta recibe solo los suyos. Lo que no se indique (la
semilla y las fechas) se toma de `mi_bici_experiment/config.py`.

`--env-manager uv` crea el entorno de `python_env.yaml` con [uv](https://docs.astral.sh/uv/); sin esa
opción, MLflow usa pyenv para instalar Python 3.14.7. `--env-manager local` usa el entorno activo.

### Ingesta de datos

Descarga los CSV faltantes de MiBici, genera el Parquet de viajes en `data/interim/` y actualiza
`data_manifest.json` con el conteo de registros y las huellas SHA-256 de los archivos.

```bash
mlflow run . -e ingest --run-name ingest --env-manager uv
```

### Comandos para reproducir P1
```bash
# Baseline (regresión lineal)
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p1 -P model=linear --env-manager uv

# Búsqueda de hiperparámetros de LightGBM: 3 combinaciones de config.py (nested runs; elige el mejor modelo evaluando val_mae) y registra el mejor modelo como champion en el Model Registry
mlflow run . --experiment-name mibici-equipo12 -e search -P question=p1 -P model=lgbm --env-manager uv

# Hiperparámetros a elección: solo cambian los que se indiquen, el resto toma los valores de modelo_3 de config.py
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p1 -P model=lgbm -P n_estimators=300 -P learning_rate=0.03 --env-manager uv

# Reproducir el mejor modelo (modelo_3 de config.py)
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p1 -P model=lgbm -P n_estimators=400 -P learning_rate=0.05 -P num_leaves=31 --env-manager uv
```

### Comandos para reproducir P2
```bash
# Baseline (regresión logística)
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p2 -P model=logistic_regression --env-manager uv

# Búsqueda de hiperparámetros de Random forest: 3 combinaciones de config.py (nested runs; elige el mejor modelo evaluando f1-score) y registra el mejor modelo como champion en el Model Registry
mlflow run . --experiment-name mibici-equipo12 -e search -P question=p2 -P model=random_forest --env-manager uv

# Hiperparámetros a elección: solo cambian los que se indiquen, el resto toma los valores de modelo_3 de config.py
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p2 -P model=random_forest -P max_depth=16 -P min_samples_leaf=5 --env-manager uv

# Reproducir el mejor modelo (modelo_3 de config.py)
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p2 -P model=random_forest -P max_depth=24 -P max_features=0.05 --env-manager uv
```

### Comandos para reproducir P3

```bash
# Baseline de la entrega anterior
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p3 -P model=kmeans -P k=4 -P stage=baseline --env-manager uv

# Búsqueda k = 3..8, PCA 0 o 2 (nested runs; elige por ARI de validación)
mlflow run . --experiment-name mibici-equipo12 -e search -P question=p3 -P model=kmeans --env-manager uv

# Candidatas
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p3 -P model=kmeans -P k=3 -P pca=2 --env-manager uv
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p3 -P model=gmm -P k=4 -P cov=spherical --env-manager uv
mlflow run . --experiment-name mibici-equipo12 -e train -P question=p3 -P model=aglomerativo -P k=3 -P linkage=ward --env-manager uv

# Model Registry: el modelo elegido como champion y el baseline como challenger
mlflow run . --experiment-name mibici-equipo12 -e register -P question=p3 -P run_id=<run_id de p3-kmeans-k3-pca2> -P alias=champion --env-manager uv
mlflow run . --experiment-name mibici-equipo12 -e register -P question=p3 -P run_id=<run_id de p3-kmeans-k4> -P alias=challenger --env-manager uv
```

### Model Registry: aliases

En Model Registry P3 se registra como `mibici-equipo12-p3-grupos` (P1 y P2 seguirán `mibici-equipo12-p1-demanda` y
`mibici-equipo12-p2-riesgo`).

Se utilizó el esquema de aliases de MLflow porque permite gestionar las versiones de los modelos de manera más flexible y eficiente, además de reemplazar el esquema tradicional de etapas (Staging y Production), obsoleto desde MLflow 2.9. Los aliases permiten asignar nombres a versiones específicas y actualizarlos sin necesidad de volver a registrar el modelo, facilitando su promoción y reemplazo. En este esquema, champion identifica la versión utilizada en producción, mientras que challenger representa una versión candidata para sustituirla. Además, cada versión conserva su trazabilidad mediante etiquetas que registran el experimento, la ejecución de origen, el commit de Git y la versión de los datos, junto con las métricas utilizadas para su selección. Esto facilita el versionado, la reproducibilidad y el despliegue controlado de modelos dentro del flujo de MLOps.

#### Código muestra para ejecutar el modelo champion de P1
```python
import mlflow
from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionLGBM

datos = RegressionLGBM()
datos.preparar_datos()

modelo = mlflow.pyfunc.load_model("models:/mibici-equipo12-p1-demanda@champion")
viajes = modelo.predict(datos.prueba[datos.todas_var])
```

#### Código muestra para ejecutar el modelo champion de P2
```python
import mlflow
import pandas as pd
from mi_bici_experiment.config import P2_DATASET_PARQUET
from mi_bici_experiment.modeling.modules.p2_classification.models import ClassificationModel

dataset_ml = pd.read_parquet(P2_DATASET_PARQUET)
datos = ClassificationModel(dataset_ml)
_, _, prueba = datos.separar_datos()

modelo = mlflow.pyfunc.load_model(
    "models:/mibici-equipo12-p2-riesgo@champion"
)
X = prueba[datos.features].astype("float64")
predicciones = modelo.predict(X)

# Map class IDs to their meanings, if desired
estados = pd.Series(predicciones, index=prueba.index).map({
    0: "Sin bicicletas",
    1: "Balanceado",
    2: "Sin anclajes",
})
```

#### Código muestra para ejecutar el modelo champion de P3
```python
import mlflow
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo12-p3-grupos@champion")
grupos = modelo.predict(perfil[["prop_fin_semana", "log_viajes_dia", "balance"]])
```

## Pruebas

Las pruebas del proyecto se encuentran en `mi-bici-experiment/tests/` y usan pytest:

- `test_dataset.py` comprueba la carga y preparación de viajes, las descargas y el manifiesto de datos.
- `test_features.py` cubre las funciones de `features.py` de P1, P2 y P3 (rezagos, promedios móviles,
  balance, umbrales de riesgo y perfiles de estación).
- `test_data_validation.py` prueba el esquema y el muestreo que usa la validación con Deepchecks.
- `test_p1.py` prueba P1: variables sin fuga de datos, partición cronológica, baseline y LightGBM,
  métricas y reproducibilidad del entrenamiento con la configuración real.
- `test_p2.py` prueba el baseline y el Random Forest de P2.
- `test_p3.py` cubre P3: perfiles de estaciones sin datos posteriores, partición temporal, los cuatro
  modelos de clustering, la configuración y la validación de perfiles. Incluye:
  - **Reproducibilidad del entrenamiento:** con la misma semilla, dos ajustes de cada modelo dan las
    mismas etiquetas, predicciones y métricas; `entrenar` (la función que usa `mlflow run`) registra
    las mismas métricas y parámetros en dos corridas de MLflow; la búsqueda con nested runs recupera
    los grupos sintéticos.
  - **Casos límite:** periodo sin viajes, umbral `min_viajes`, estaciones que solo salen en fin de
    semana o solo reciben, ninguna estación común entre periodos, más grupos que estaciones y perfiles
    con nulos.
- `test_seguridad.py` revisa que Git no versione `.env`, bases de MLflow, modelos ni datos de MiBici,
  que `.gitignore` los excluya y que ningún archivo versionado contenga tokens, llaves o contraseñas.

Las pruebas usan datos sintéticos, no necesitan el Parquet y las de MLflow escriben en un directorio
temporal, así que corren en un clon limpio y en GitHub Actions (`.github/workflows/mlflow.yml`).

Para ejecutar todas las pruebas desde la raíz del repositorio en PowerShell o Bash:

1. Activa el entorno virtual de Python:

   ```bash
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

   Si creaste el entorno dentro de `mi-bici-experiment/` como se indica en la instalación, usa `.\mi-bici-experiment\.venv\Scripts\Activate.ps1` en su lugar.

2. Ejecuta todas las pruebas:

   ```powershell
   python -m pytest mi-bici-experiment/tests -v
   ```

La opción `-v` muestra el resultado de cada prueba. Para correr solo un grupo, por ejemplo la
reproducibilidad de P3: `python -m pytest mi-bici-experiment/tests/test_p3.py -k "semilla or reproducible" -v`.

![Ejecución de pytest](mi-bici-experiment/docs/assets/pytest_execution.png)

## Validación del Parquet con Deepchecks

El archivo `data/interim/trips_10_2024_to_09_2026.parquet` se valida con
`mi_bici_experiment.data_validation`. Deepchecks 0.19.1 necesita un entorno
separado: el entorno principal usa Python 3.14, pandas 3 y scikit-learn 1.9,
con los que esta versión de Deepchecks no carga.

Desde la raíz del repositorio en PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\mi-bici-experiment\run_deepchecks.ps1
```

![Ejecución de la validación con Deepchecks](mi-bici-experiment/docs/assets/deep_checks_execution.png)

El script usa `uv` del entorno principal `.venv` (o de `PATH`). Si falta
Python 3.10, `uv` lo descarga. Crea `mi-bici-experiment/.venv-deepchecks-windows` una sola vez;
en ejecuciones posteriores conserva el entorno, instala los paquetes de
`requirements-deepchecks.txt`, ejecuta `tests/test_data_validation.py` y
después valida el Parquet con Deepchecks.

En Linux, con `uv` instalado y disponible en `PATH` (o en `.venv/bin/uv`),
ejecutar desde la raíz del repositorio:

```bash
bash mi-bici-experiment/run_deepchecks.sh
```

La versión de Bash crea `mi-bici-experiment/.venv-deepchecks-linux` para que
pueda coexistir con el entorno virtual de Windows en el mismo repositorio.

La herramienta revisa el esquema y cuenta los nulos en **todas** las filas.
Después toma una muestra reproducible y acotada de cada periodo para que
Deepchecks analice tipos mixtos, nulos, filas repetidas y drift de IDs de
estación, coordenadas y capacidad. Los umbrales iniciales son 0% de nulos,
1% de filas repetidas y 0.2 de drift por variable; se deben ajustar según
el uso del modelo. Por defecto, entrenamiento termina antes
del 1 de julio de 2026 y prueba empieza ese día (periodos de P3). Se puede
modificar con `--cutoff YYYY-MM-DD` y `--sample-size N`, así como proporcionar
otro archivo con `--parquet RUTA`.

Los resultados quedan en `mi-bici-experiment/reports/deepchecks/`: `deepchecks.html`,
`deepchecks.json` y `audit.json`. Para abrir el reporte HTML en el navegador predeterminado,
ejecuta desde la raíz del repositorio:

```powershell
Start-Process .\mi-bici-experiment\reports\deepchecks\deepchecks.html
```

En Linux, usa:

```bash
xdg-open mi-bici-experiment/reports/deepchecks/deepchecks.html
```

![Reporte HTML generado por Deepchecks](mi-bici-experiment/docs/assets/deep_check_report.png)

El comando devuelve código 1 cuando hay nulos o falla una condición de Deepchecks. Revisa las filas
repetidas antes de eliminarlas: el Parquet conserva fechas truncadas al día y no incluye un ID único
de viaje, por lo que viajes distintos pueden verse iguales.

### Validación de los perfiles de P3

P3 no recibe viajes sino una fila por estación (`prop_fin_semana`, `log_viajes_dia`, `balance`).
[`validacion_p3.py`](mi-bici-experiment/mi_bici_experiment/validacion_p3.py) valida esa tabla en
entrenamiento (2025-07 a 2026-06) y en prueba (2026-07 a 2026-09). Son dos pasos porque los perfiles
se construyen con `features.py` (entorno principal) y Deepchecks corre en Python 3.10:

```bash
python -m mi_bici_experiment.validacion_p3 exportar   # desde mi-bici-experiment/, entorno principal
bash mi-bici-experiment/run_deepchecks.sh              # incluye el paso `validar` de P3
```

| Chequeo | Herramienta | Resultado (datos 2024-10 a 2026-09) |
|---------|-------------|-------------------------------------|
| Nulos, tipos mixtos, valor único, perfiles duplicados, outliers | Deepchecks (train y test) | Aprobado |
| IDs de estación únicos y rangos (`prop_fin_semana` ∈ [0, 1], `balance` ∈ [−1, 1], `log_viajes_dia` ≥ 0) | `revisar_perfiles` | Aprobado |
| Drift por variable (Kolmogorov-Smirnov) | Deepchecks `FeatureDrift` | 0.107, 0.076 y 0.169 (< 0.2) |
| Drift multivariado (clasificador de dominio) | Deepchecks `MultivariateDrift` | **0.394 > 0.2**: falla |
| Mismo drift solo con estaciones que ya existían | Deepchecks | 0.19 (< 0.2) |
| Estaciones nuevas en prueba | `revisar_perfiles` | **79** de 454, sin grupo de entrenamiento |

Hallazgo: el drift multivariado viene de las 79 estaciones abiertas después del entrenamiento; con las
375 estaciones comunes baja a 0.19. Por eso el ARI de estabilidad de P3 se mide solo con estaciones
comunes, y las nuevas se asignan con `predict` al grupo más cercano. `balance` es la variable que más
cambia (KS ≈ 0.17) entre el año de entrenamiento y el trimestre de julio a septiembre.

El script termina con código 1 si falla la integridad. El drift se reporta en
`reports/deepchecks/p3/` (`integridad_train.html`, `integridad_test.html`, `drift.html`,
`drift_estaciones_comunes.html` y `audit_p3.json`) sin detener el proceso: un cambio estacional
es esperable y es lo que mide el ARI de estabilidad.

![Validación de los perfiles de P3 con Deepchecks](mi-bici-experiment/docs/assets/deepchecks_p3_execution.png)
![Reporte de drift de P3](mi-bici-experiment/docs/assets/deepchecks_p3_drift.png)

## Análisis de seguridad con Bandit y pip-audit

Las dependencias del proyecto incluyen Bandit y pip-audit. Activa el entorno virtual en el que instalaste `mi-bici-experiment/requirements.txt` y ejecuta los comandos desde la raíz del repositorio.

### Bandit

Bandit analiza el código Python y señala patrones potencialmente inseguros. Este comando escanea el repositorio y excluye los entornos virtuales `.venv` y `venv`:

```powershell
python -m bandit -r . -x .venv,venv
```

![Resultado de un análisis Bandit](mi-bici-experiment/docs/assets/bandit_execution.png)

### pip-audit

pip-audit revisa los paquetes instalados en el entorno virtual activo y reporta vulnerabilidades conocidas:

```powershell
python -m pip_audit
```

![Resultado de un análisis pip-audit](mi-bici-experiment/docs/assets/pip_audit_execution.png)

Revisa los hallazgos y actualiza las versiones afectadas antes de integrar cambios. Los resultados pueden variar según las versiones instaladas y las vulnerabilidades registradas al momento de ejecutar las herramientas.

### Hallazgos y correcciones

Para auditar las versiones fijadas (no solo el entorno activo):

```bash
python -m pip_audit -r mi-bici-experiment/requirements.txt
```

El 2026-10-29 pip-audit reportó 3 vulnerabilidades, corregidas subiendo las versiones fijadas:

| Paquete | Versión | Vulnerabilidad | Corrección |
|---------|---------|----------------|------------|
| certifi | 2026.7.22 | PYSEC-2026-4320: incluye una CA raíz con intermedias comprometidas | 2026.10.19 |
| mlflow (y mlflow_skinny, mlflow_tracing) | 3.17.0 | CVE-2026-53995: path traversal en el endpoint de trazas del servidor de tracking | 3.17.2 |

Después de la corrección: `No known vulnerabilities found`.

![pip-audit después de la corrección](mi-bici-experiment/docs/assets/pip_audit_requirements.png)

Bandit sobre `mi_bici_experiment/` no reporta hallazgos de severidad alta. Los de severidad baja o media
están revisados:

| Hallazgo | Ubicación | Por qué se acepta |
|----------|-----------|-------------------|
| B310 `urlopen` (media) | `dataset.py` | Solo abre las URL `https://` fijas de `config.py`, no entradas del usuario |
| B404, B603, B607 `subprocess` (baja) | `tracking.py` | Ejecuta `git` con argumentos fijos y sin `shell=True` para obtener el commit |

### Credenciales y datos sensibles

- Las variables de entorno se leen de `mi-bici-experiment/.env` (ignorado por Git) con python-dotenv.
  [`.env.example`](mi-bici-experiment/.env.example) lista las que usa el proyecto (`MLFLOW_TRACKING_URI`
  y, si el servidor tiene autenticación, `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`) sin
  valores secretos: `cp mi-bici-experiment/.env.example mi-bici-experiment/.env`.
- `.gitignore` excluye `.env`, `data/`, `mlflow.db`, `mlruns/`, `mlartifacts/`, modelos serializados y los
  reportes de Deepchecks. `tests/test_seguridad.py` lo comprueba en cada `pytest` y en CI.
- Los datos de MiBici no incluyen nombres ni identificadores personales; a MLflow solo se registran
  agregados por estación y la versión (hash) de los datos.
- Un `mlflow.db` se subió por error en c8b88f2 y se quitó en 6a47b0d. Sigue en el historial; se revisó
  y solo tiene corridas de experimentos (con el usuario del sistema y rutas locales en las etiquetas de
  MLflow), sin credenciales (la tabla `secrets` está vacía) ni datos de usuarios de MiBici.

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
| Julián Hazael Martínez Hernández | [@julianhmh01](https://github.com/julianhmh01) | P1 (regresión) y documentación |
| Eduardo Sebastián Vela Alanís | [@eduardoVela2022](https://github.com/eduardoVela2022) | EDA, ML Canvas y P2 (clasificación)  |
| Jordan Piero Borda Colque | [@neronapier](https://github.com/neronapier) | P3 (agrupación) y documentación

## Referencias

- Dorard, L. (2018, 9 de marzo). *From data to AI with the Machine Learning Canvas (Part I)*. Own Machine Learning.
- Huyen, C. (2022). *Designing machine learning systems*. O'Reilly Media.
- MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. https://www.mibici.net/es/datos-abiertos/
- MLflow. (s.f.-a). *MLflow Model Registry*. https://mlflow.org/docs/latest/ml/model-registry/
- MLflow. (s.f.-b). *MLflow Projects*. https://mlflow.org/docs/latest/ml/projects/
- MLflow. (s.f.-c). *MLflow Tracking*. https://mlflow.org/docs/latest/ml/tracking/
- Treveil, M., et al. (2020). *Introducing MLOps*. O'Reilly Media.
- Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.
