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
Mac
6.1 python3 mi-bici-experiment/scripts/entrenar_modelos.py
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

## P2: búsqueda de hiperparámetros de Random Forest

Desde `mi-bici-experiment/`, sin opciones se ejecuta el baseline logístico y se comparan las
configuraciones de `P2_COMBINACIONES` en `config.py`:

```bash
python -m mi_bici_experiment.modeling.modules.p2_classification.models
```

Si se proporciona al menos un hiperparámetro, se entrena únicamente un Random Forest personalizado.
Los valores omitidos se toman de `P2_COMBINACIONES["modelo_3"]`:

```bash
python -m mi_bici_experiment.modeling.modules.p2_classification.models \
  --n-estimators 300 --max-depth 16 --min-samples-leaf 5
```

Opciones: `--n-estimators`, `--max-depth`, `--min-samples-split`, `--min-samples-leaf`,
`--max-features` (`sqrt`, `log2` o una fracción entre 0 y 1), `--bootstrap/--no-bootstrap`,
`--max-samples`, `--class-weight` (`balanced`, `balanced_subsample` o `none`),
`--random-state` y `--n-jobs`. `--no-bootstrap` desactiva el muestreo por árbol;
no se combina con `--max-samples`. La lista completa está en `--help`.

La búsqueda registra un run padre `P2-random-forest-search` y un hijo por configuración con
sus parámetros, métricas y artefactos. El mejor candidato se elige por F1 macro de validación,
sin usar prueba para seleccionar. Su identificador queda en `best_run_id` del padre y el
resumen de candidatos en `busqueda/resultados.json`.

Al terminar la búsqueda, el pipeline del Random Forest ganador se registra como
`mibici-equipo12-p2-riesgo` (`P2_NOMBRE_REGISTRO` en `config.py`) y recibe el alias
`champion`. La versión conserva el run de origen, la combinación, el F1 de validación,
la versión de datos y los metadatos Git del entrenamiento ganador. Los candidatos
permanecen como artefactos en sus runs; solo el ganador de la búsqueda se registra bajo
este nombre. Su URI es `models:/mibici-equipo12-p2-riesgo@champion`.

## P3: grupos de estaciones con patrones de uso similares

Pregunta: ¿qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?
Usuario: planeación urbana y expansión del sistema.

El código está en `mi_bici_experiment/modeling/modules/p3_clustering/` y se ejecuta con `mlflow run`
desde la raíz del repositorio. Los comandos, los artefactos, los resultados y el esquema del Model
Registry están en la sección "MLflow Project (actividad 5)" del [README principal](../README.md).

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

