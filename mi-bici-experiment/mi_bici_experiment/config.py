from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

##   Regresión (P1)   ##
RANDOM_STATE = 42
FECHA_INICIO = "2025-10-01"
TEST_PARTICION = 0.15
VAL_PARTICION = 0.15
FECHA_FIN = "2026-09-30"
STATION_COL = "origen_id"
FECHA_INICIO_COL = "inicio_del_viaje"
OBJETIVO_COL = "trips"
LAGS = (1, 7)  # viajes del día anterior y de hace 7 días
PM_DIAS = 7  # promedio móvil de los últimos 7 días
TRIPS_PARQUET = INTERIM_DATA_DIR / "trips_10_2024_to_09_2026.parquet"

##  MLflow   ##
MLFLOW_EXPERIMENT = "mibici-equipo12"

MI_BICI_CSVS = [
    "https://mibici.net/site/assets/files/91758/datos_abiertos_2024_10.csv",
    "https://mibici.net/site/assets/files/93516/datos_abiertos_2024_11.csv",
    "https://mibici.net/site/assets/files/94962/datos_abiertos_2024_12.csv",
    "https://mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv",
    "https://mibici.net/site/assets/files/99879/datos_abiertos_2025_02.csv",
    "https://mibici.net/site/assets/files/101986/datos_abiertos_2025_03.csv",
    "https://mibici.net/site/assets/files/103813/datos_abiertos_2025_04.csv",
    "https://mibici.net/site/assets/files/105513/datos_abiertos_2025_05.csv",
    "https://mibici.net/site/assets/files/107088/datos_abiertos_2025_06.csv",
    "https://mibici.net/site/assets/files/108898/datos_abiertos_2025_07.csv",
    "https://mibici.net/site/assets/files/111021/datos_abiertos_2025_08.csv",
    "https://mibici.net/site/assets/files/113216/datos_abiertos_2025_09.csv",
    "https://mibici.net/site/assets/files/115104/datos_abiertos_2025_10.csv",
    "https://mibici.net/site/assets/files/116773/datos_abiertos_2025_11.csv",
    "https://mibici.net/site/assets/files/118153/datos_abiertos_2025_12.csv",
    "https://mibici.net/site/assets/files/120163/datos_abiertos_2026_01.csv",
    "https://mibici.net/site/assets/files/121761/datos_abiertos_2026_02.csv",
    "https://mibici.net/site/assets/files/123236/datos_abiertos_2026_03.csv",
    "https://mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "https://mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv",
    "https://mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv",
    "https://mibici.net/site/assets/files/129006/datos_abiertos_2026_07.csv",
    "https://mibici.net/site/assets/files/130362/datos_abiertos_2026_08.csv",
    "https://mibici.net/site/assets/files/131825/datos_abiertos_2026_09.csv",
    "https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv",
]

##   Columnas y archivos que usan P2 y P3 (las generales están en la sección de P1)   ##
EQUIPO = "12"
NOMENCLATURA_CSV = RAW_DATA_DIR / "nomenclatura_2026_09.csv"
DESTINO_COL = "destino_id"
FECHA_FIN_COL = "fin_del_viaje"

##   Clustering (P3)   ##
# Entrenamiento de 12 meses (2025-07 a 2026-06, un ciclo estacional completo); la validación son
# sus 3 últimos meses (ajuste hasta 2026-03) y la prueba de estabilidad, 2026-07 a FECHA_FIN.
P3_INICIO_ENTRENAMIENTO = "2025-07-01"
P3_FIN_ENTRENAMIENTO = "2026-06-30"
P3_MESES_VALIDACION = 3
P3_MIN_PROP_GRUPO = 0.05  # grupos con menos del 5 % de las estaciones no son accionables
P3_MIN_VIAJES = 300  # salidas mínimas en entrenamiento para que el perfil sea estable
P3_K_BASELINE = 4
P3_K_CANDIDATO = 3  # modelo elegido en la actividad 5 (k-means + PCA, ver README)
P3_K_CANDIDATOS = [3, 4, 5, 6, 7, 8]
P3_PCA_COMPONENTES = 2
P3_N_INIT = 20  # inicializaciones de k-means; se queda la de menor inercia
P3_FIGURES_DIR = FIGURES_DIR / "p3"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
