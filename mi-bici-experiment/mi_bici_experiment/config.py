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
FECHA_INICIO = "2025-10-01"   # con solo septiembre usa "2026-09-01"
FECHA_FIN = "2026-09-30"
STATION_COL = "Origen_Id"
FECHA_INICIO_COL = "Inicio_del_viaje"
OBJETIVO_COL = "trips"
LAGS = (1, 7)  # viajes del día anterior y de hace 7 días
PM_DIAS = 7  # promedio móvil de los últimos 7 días
TEST_PARTICION = 0.2  # último 20% de las fechas = test
FEATURES_PARQUET = INTERIM_DATA_DIR/"features.parquet"

##  MLflow   ##
MLFLOW_EXPERIMENT = "mibici-equipo12"

MI_BICI_CSVS = [
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
    "https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
]



# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass