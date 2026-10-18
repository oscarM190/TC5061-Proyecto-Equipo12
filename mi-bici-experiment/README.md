Creacion de modelos para reponder preguntas de negocio de miBici

¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro?
¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro?
¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?

Execute the commands:
1.- virtualenv venv3.13 --python=python3.14.7
2.- source venv3.13/bin/activate
3.- cd mi-bici-experiment
Mac
4.1.- pip3 install -r requirements.txt
Windows
4.2.- pip install -r requirements.txt
Mac
5.1- python3 -m mi_bici_experiment.dataset
Windows
5.1- python -m mi_bici_experiment.features

## Descargar datos de MiBici en local

Desde la raíz del repositorio, instalar los certificados de confianza y ejecutar:

```bash
python3 -m pip install --upgrade certifi
python3 mi-bici-experiment/scripts/download_mibici.py
```

Descarga los 142 CSV de diciembre de 2014 a septiembre de 2026 listados en
`scripts/mibici_urls.txt` en `mi-bici-experiment/downloads/`, conservando sus nombres.
La carpeta se crea automáticamente y está excluida de Git. Los archivos existentes
que no estén vacíos se omiten; usar `--overwrite` para descargarlos de nuevo o
`--output-dir RUTA` para elegir otra carpeta.

Las descargas usan archivos temporales `.part` y hasta tres intentos por archivo.
Si una URL falla, el script continúa con las demás y termina con código 1;
se puede volver a ejecutar para descargar los archivos pendientes.

El script conserva la verificación HTTPS y agrega los certificados de `certifi`
si está instalado, además de los certificados disponibles en Python. Esto permite
usar instalaciones de Python en macOS sin un almacén de certificados configurado.
Si tu red utiliza una CA privada, puedes agregar su archivo PEM con
`--ca-file /ruta/certificados.pem`.

## P3: grupos de estaciones con patrones de uso similares

Pregunta: ¿qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?
Usuario: planeación urbana y expansión del sistema.

| Elemento | Definición |
|----------|------------|
| Qué se agrupa | Estaciones (una fila por estación) |
| Variables | Proporción de salidas y de llegadas por hora en días hábiles (48), peso del fin de semana, log de viajes por día y balance llegadas/salidas (`features.py`, sección P3) |
| Periodo | Ajuste con 2025-10 a 2026-06; 2026-07 a 2026-09 solo mide estabilidad (`P3_FIN_ENTRENAMIENTO` en `config.py`) |
| Limpieza | Sin duraciones no positivas o mayores a 6 h ni devoluciones en la misma estación en menos de 60 s; solo estaciones con al menos 300 salidas |
| Baseline | K-means con k = 4 sobre el perfil horario de salidas |
| Candidato | Estandarización + PCA (3) + k-means, k entre 3 y 8 elegido por silueta |
| Métricas | `train_silhouette`, `train_davies_bouldin` y `train_calinski_harabasz` en el espacio de cada modelo; `train_silhouette_espacio_comun` y `train_davies_bouldin_espacio_comun` sobre el perfil completo estandarizado (las únicas comparables entre baseline y candidato); `test_ari_estabilidad` (grupos de entrenamiento frente a los asignados con el perfil posterior) |

```bash
python -m mi_bici_experiment.dataset      # descarga y Parquet (Oscar)
python -m mi_bici_experiment.features     # data/interim/features.parquet
python -m mi_bici_experiment.modeling.train clustering-p3
python -m pytest tests
```

Cada run se registra en el experimento `mibici-equipo12` con parámetros de reproducción (fechas,
`data_version` = huella SHA-256 de los CSV crudos, semilla e hiperparámetros), etiquetas (`equipo`,
`pregunta`, `tipo_modelo`, `git_commit`, `git_branch`, `git_dirty`) y artefactos: mapa de grupos,
perfil horario por grupo, silueta por k, `estaciones_por_grupo.csv` y `resumen_grupos.csv`.

Resultados (2025-10 a 2026-09; 378 estaciones con al menos 300 salidas en entrenamiento):

| Run | k | train_silhouette_espacio_comun | train_davies_bouldin_espacio_comun | test_ari_estabilidad |
|-----|---|--------------------------------|------------------------------------|----------------------|
| `P3-kmeans-baseline-k4` | 4 | 0.038 | 3.322 | 0.462 |
| `P3-pca-kmeans-candidato` | 3 | **0.073** | **2.780** | **0.615** |

`train_silhouette` (0.111 frente a 0.275) no se compara entre runs: el baseline la mide en las 24
proporciones de salidas y el candidato en los 3 componentes de PCA. El sistema pasa de 368 estaciones
activas en 2025-10 a 468 en 2026-09 (las nuevas abren desde 2026-06), así que la estabilidad se mide
solo con las estaciones que existen en ambos periodos.

Grupos del candidato: origen matutino (171 estaciones, periferia norte y sureste, salen a las 8 h y
llegan a las 18 h), destino matutino (111, poniente central, llegan a las 8 h y tienden a vaciarse) y
uso repartido en el día (95, centro). La exploración está en
`notebooks/1.0-jpbc-p3-clustering-estaciones.ipynb`.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mi_bici_experiment and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mi_bici_experiment   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mi_bici_experiment a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

