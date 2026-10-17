from pathlib import Path

from loguru import logger
import typer
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from mi_bici_experiment.config import INTERIM_DATA_DIR, LAGS, PM_DIAS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mi_bici_experiment.plots import print_balance_normalizado

app = typer.Typer()

def limpiar_viajes(viajes: pd.DataFrame, col_estacion: str, col_fecha: str) -> pd.DataFrame:
    """Quita registros sin estación o con fecha inválida y se muestra cuántos se quitaron."""
    antes = len(viajes)
    viajes = viajes.copy()
    viajes[col_fecha] = pd.to_datetime(viajes[col_fecha], errors="coerce")
    viajes = viajes.dropna(subset=[col_estacion, col_fecha])
    logger.info(f"Registros descartados por nulos o fechas inválidas: {antes-len(viajes):,}")
    return viajes


def construir_viajes_diarios(viajes: pd.DataFrame, col_estacion: str, col_fecha: str) -> pd.DataFrame:
    """Cuenta viajes iniciados por estación y día.
    Rellena con 0 los días sin viajes a partir del primer día con actividad de cada estación."""

    df= viajes[[col_estacion, col_fecha]].copy()
    df["date"] = df[col_fecha].dt.normalize()

    diario = df.groupby([col_estacion, "date"]).size().rename("trips")
    indice_completo = pd.MultiIndex.from_product(
        [diario.index.get_level_values(0).unique(),
         pd.date_range(df["date"].min(), df["date"].max(), freq="D")],
        names=[col_estacion, "date"])
    diario=diario.reindex(indice_completo, fill_value=0).reset_index()

    primer_dia= diario["date"].where(diario["trips"] > 0).groupby(diario[col_estacion]).transform("min")
    return diario[diario["date"]>= primer_dia].reset_index(drop=True)


def agregar_variables(diario: pd.DataFrame, col_estacion: str) -> pd.DataFrame:
    """Agrega variables de calendario e históricas.

    Calendario: día de la semana, mes y fin de semana (para la estacionalidad).
    Históricas: viajes del día anterior, de hace 7 días y promedio móvil de los 7 días previos.

    Todo se calcula con `shift` por estación, para que no haya data leakage: el valor del día t usa solo días anteriores a t.
    Los primeros días de cada estación no tienen historial, por ende, se descartan.
    """
    diario = diario.sort_values([col_estacion, "date"]).copy()
    por_estacion = diario.groupby(col_estacion)["trips"]

    # Viajes de días anteriores
    diario["trips_lag_1"] = por_estacion.shift(1)
    diario["trips_lag_7"] = por_estacion.shift(7)

    #Promedio móvil de los 7 días previos (shift(1) evita usar el día actual)
    previos = por_estacion.shift(1)
    promedio = previos.groupby(diario[col_estacion]).rolling(7).mean()
    diario["trips_roll_mean_7"] = promedio.reset_index(level=0, drop=True)

    diario["dayofweek"] = diario["date"].dt.dayofweek
    diario["month"] = diario["date"].dt.month
    diario["is_weekend"] = (diario["dayofweek"] >= 5).astype(int)

    # Se descartan los primeros días de cada estación
    columnas_historicas = ["trips_lag_1", "trips_lag_7", "trips_roll_mean_7"]
    return diario.dropna(subset=columnas_historicas).reset_index(drop=True)

def clasificacion_variable_objetivo_y_caracterizticas():
    print("TODO")

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
):
    trips_df = pd.read_parquet(input_path / "trips.parquet")
    nomenclatura_df = pd.read_parquet(input_path / "nomenclatura.parquet")

    features_df = trips_df[
        ["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
    ].merge(
        nomenclatura_df[["id", "latitude", "longitude", "dpcapacity"]].rename(
            columns={"id": "Origen_Id"}
        ),
        on="Origen_Id",
        how="left",
        validate="many_to_one",
    )

    print(features_df.info())

    output_path.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path / "features.parquet", index=False)


def agregar_balance_neto_y_normalizado(df):
    df['Fecha'] = df['Inicio_del_viaje'].dt.floor('d')

    salidas = df.groupby(['Fecha', 'Origen_Id']).size().reset_index(name='Salidas')
    salidas.rename(columns={'Origen_Id': 'Estacion_Id'}, inplace=True)

    llegadas = df.groupby(['Fecha', 'Destino_Id']).size().reset_index(name='Llegadas')
    llegadas.rename(columns={'Destino_Id': 'Estacion_Id'}, inplace=True)

    capacidad = df.groupby('Origen_Id')['dpcapacity'].max().reset_index()
    capacidad.rename(columns={'Origen_Id': 'Estacion_Id'}, inplace=True)

    # Unir Salidas y Llegadas
    balance_diario = pd.merge(salidas, llegadas, on=['Fecha', 'Estacion_Id'], how='outer').fillna(0)

    # Unir la capacidad de la estación
    balance_diario = pd.merge(balance_diario, capacidad, on='Estacion_Id', how='left')

    # Evitar división por cero si alguna estación tiene capacidad 0 o NaN
    balance_diario['dpcapacity'] = balance_diario['dpcapacity'].replace(0, np.nan)

    # Calcular el Balance Neto y el Balance Normalizado
    balance_diario['Balance_Neto'] = balance_diario['Llegadas'] - balance_diario['Salidas']
    balance_diario['Balance_Normalizado'] = balance_diario['Balance_Neto'] / balance_diario['dpcapacity']

    # Limpiar valores nulos resultantes de capacidades inválidas
    balance_diario = balance_diario.dropna(subset=['Balance_Normalizado'])
    print_balance_normalizado(balance_diario)
    return balance_diario

def clasificacion_variable_objetivo_y_caracterizticas(balance_diario):
    balance_diario = balance_diario[(balance_diario['Balance_Normalizado'] >= -1.5) & 
                                (balance_diario['Balance_Normalizado'] <= 1.5)].copy()

    #  Creación de la variable actual de estado (0: Sin bicis, 1: Balanceado, 2: Sin anclajes)
    condiciones = [
        (balance_diario['Balance_Normalizado'] <= -0.70),
        (balance_diario['Balance_Normalizado'] >= 0.70)
    ]
    opciones = [0, 2]
    balance_diario['Estado_Actual'] = np.select(condiciones, opciones, default=1)

    # Ingeniería de Características: Variables Temporales
    balance_diario['Dia_Semana'] = balance_diario['Fecha'].dt.dayofweek
    balance_diario['Es_Fin_Semana'] = balance_diario['Dia_Semana'].isin([5, 6]).astype(int)
    balance_diario['Mes'] = balance_diario['Fecha'].dt.month

    # Construcción de la Variable Objetivo (y) para T+1
    # Ordenamos estrictamente por estación y fecha para evitar mezcla de series de tiempo
    balance_diario = balance_diario.sort_values(['Estacion_Id', 'Fecha'])

    # Desplazamos el estado un día hacia atrás dentro de cada grupo de estación
    # Esto alinea los features del "Hoy" con la clase de riesgo de "Mañana"
    balance_diario['Riesgo_T1'] = balance_diario.groupby('Estacion_Id')['Estado_Actual'].shift(-1)

    # El último día histórico de cada estación queda nulo, lo descartamos para el entrenamiento
    dataset_ml = balance_diario.dropna(subset=['Riesgo_T1']).copy()
    dataset_ml['Riesgo_T1'] = dataset_ml['Riesgo_T1'].astype(int)
    return dataset_ml


def creacion_variables_memoria_historica(dataset_ml):
    # Asegurar de que el dataset esté perfectamente ordenado
    dataset_ml = dataset_ml.sort_values(['Estacion_Id', 'Fecha'])

    # Variables de Retardo (Lags)
    # ¿Cómo se comportó el balance ayer y antier?
    dataset_ml['Balance_Lag_1'] = dataset_ml.groupby('Estacion_Id')['Balance_Normalizado'].shift(1)
    dataset_ml['Balance_Lag_2'] = dataset_ml.groupby('Estacion_Id')['Balance_Normalizado'].shift(2)

    # ¿Cómo se comportó el balance el mismo día de la semana pasada?
    dataset_ml['Balance_Lag_7'] = dataset_ml.groupby('Estacion_Id')['Balance_Normalizado'].shift(7)

    # Estado categórico previo (para capturar transiciones de estado)
    dataset_ml['Estado_Lag_1'] = dataset_ml.groupby('Estacion_Id')['Estado_Actual'].shift(1)
    dataset_ml['Estado_Lag_7'] = dataset_ml.groupby('Estacion_Id')['Estado_Actual'].shift(7)

    # Ventanas Móviles (Rolling Windows)
    # Promedio del balance en los últimos 3 y 7 días (captura la tendencia suavizada)
    dataset_ml['Media_Movil_3d'] = dataset_ml.groupby('Estacion_Id')['Balance_Normalizado'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    dataset_ml['Media_Movil_7d'] = dataset_ml.groupby('Estacion_Id')['Balance_Normalizado'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Volatilidad en los últimos 3 días (estaciones con mucha varianza son más impredecibles)
    dataset_ml['Volatilidad_3d'] = dataset_ml.groupby('Estacion_Id')['Balance_Normalizado'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    ).fillna(0) # Llenamos con 0 donde no hay suficientes datos para desviación estándar

    #  Limpieza post-ingeniería
    # Al hacer shift(7), los primeros 7 días de cada estación quedarán con nulos en Lag_7.
    # Para entrenar modelos (como Random Forest o XGBoost), es mejor eliminar estos registros incompletos.
    dataset_ml = dataset_ml.dropna(subset=['Balance_Lag_7', 'Estado_Lag_7']).copy()

    # Convertir variables de estado a enteros tras el dropna
    dataset_ml['Estado_Lag_1'] = dataset_ml['Estado_Lag_1'].astype(int)
    dataset_ml['Estado_Lag_7'] = dataset_ml['Estado_Lag_7'].astype(int)

    print(dataset_ml[['Fecha', 'Estacion_Id', 'Balance_Lag_1', 'Media_Movil_3d', 'Riesgo_T1']].head(10))
    return dataset_ml

def creacion_variables_geoespaciales(dataset_ml, df):
    # Extraer el catálogo estático de estaciones con sus coordenadas
    # df es dataframe original crudo
    estaciones_geo = df[['Origen_Id', 'latitude', 'longitude']].drop_duplicates()
    estaciones_geo = estaciones_geo.rename(columns={'Origen_Id': 'Estacion_Id'}).dropna()
    print(dataset_ml.info())
    print(dataset_ml.info())
    # 2. Macro-arquitectura: Creación de Clústeres Urbanos (Zonas)
    # Agrupamos las estaciones en 10 zonas geográficas distintas
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    estaciones_geo['Zona_Espacial'] = kmeans.fit_predict(estaciones_geo[['latitude', 'longitude']])

    # 3. Micro-resiliencia: Densidad de red (Estaciones a menos de ~500 metros)
    # Convertimos coordenadas a radianes para usar la fórmula Haversine (distancia real en esfera)
    coords_rad = np.radians(estaciones_geo[['latitude', 'longitude']])

    # Radio de la Tierra ~6371 km. Buscamos vecinos en 0.5 km (500 metros).
    radio_radianes = 0.5 / 6371.0 
    nn = NearestNeighbors(radius=radio_radianes, metric='haversine')
    nn.fit(coords_rad)

    # Calculamos cuántas estaciones caen en ese radio (restando 1 para excluir la estación misma)
    vecinos = nn.radius_neighbors(coords_rad, return_distance=False)
    estaciones_geo['Densidad_500m'] = [len(v) - 1 for v in vecinos]

    # 4. Consolidación de la Arquitectura de Datos
    # Unimos estas características estáticas al dataset histórico de Machine Learning
    dataset_ml = pd.merge(dataset_ml, estaciones_geo, on='Estacion_Id', how='left')

    # Verificamos la matriz final
    print(dataset_ml[['Estacion_Id', 'Zona_Espacial', 'Densidad_500m', 'Balance_Lag_1', 'Riesgo_T1']].head())
    return dataset_ml


if __name__ == "__main__":
    app()
