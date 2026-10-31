from loguru import logger

import pandas as pd
import numpy as np

from mi_bici_experiment.config import (
    LAGS,
    PM_DIAS,
    DESTINO_COL,
    FECHA_FIN_COL,
    FECHA_INICIO_COL,
    P3_MIN_VIAJES,
    STATION_COL,
)

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

def agregar_balance_neto_y_normalizado(df):
    """Balance diario sobre el calendario operativo observado de cada estación.

    Sin un catálogo de altas/bajas, se infiere el periodo entre el primer y último
    movimiento (salida o llegada). Solo se completan con cero los días interiores.
    """
    salidas = (
        df.assign(Fecha=df['Inicio_del_viaje'].dt.normalize())
        .groupby(['Fecha', 'Origen_Id']).size().rename('Salidas')
        .rename_axis(['Fecha', 'Estacion_Id'])
    )
    llegadas = (
        df.assign(Fecha=df['Fin_del_viaje'].dt.normalize())
        .groupby(['Fecha', 'Destino_Id']).size().rename('Llegadas')
        .rename_axis(['Fecha', 'Estacion_Id'])
    )
    movimientos = pd.concat([salidas, llegadas], axis=1).fillna(0).reset_index()
    capacidad = df.groupby('Origen_Id')['dpcapacity'].max()
    diarios = []
    for estacion_id, station in movimientos.groupby('Estacion_Id'):
        calendario = pd.date_range(station['Fecha'].min(), station['Fecha'].max(), freq='D')
        diario = station.set_index('Fecha')[['Salidas', 'Llegadas']].reindex(
            calendario, fill_value=0
        )
        diario.index.name = 'Fecha'
        diario['Estacion_Id'] = estacion_id
        diario['dpcapacity'] = capacidad.get(estacion_id, np.nan)
        diarios.append(diario.reset_index())
    if not diarios:
        raise ValueError("No hay movimientos para construir el calendario diario de P2.")
    balance_diario = pd.concat(diarios, ignore_index=True)
    balance_diario = balance_diario.loc[
        np.isfinite(balance_diario['dpcapacity']) & balance_diario['dpcapacity'].gt(0)
    ].copy()
    balance_diario['Balance_Neto'] = balance_diario['Llegadas'] - balance_diario['Salidas']
    balance_diario['Balance_Normalizado'] = (
        balance_diario['Balance_Neto'] / balance_diario['dpcapacity']
    )
    from mi_bici_experiment.plots import print_balance_normalizado
    print_balance_normalizado(balance_diario)
    return balance_diario

def clasificacion_variable_objetivo_y_caracterizticas(balance_diario):
    """Clasifica el balance sin recortar extremos y une el estado del día T+1 exacto."""
    # Preserve extreme days and the complete calendar before computing history.
    balance_diario = balance_diario.copy()

    #  Creación de la variable actual de estado (0: Sin bicis, 1: Balanceado, 2: Sin anclajes)
    condiciones = [
        (balance_diario['Balance_Normalizado'] <= -0.45),
        (balance_diario['Balance_Normalizado'] >= 0.45)
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

    # Match the label to the exact following calendar day, never the next available row.
    balance_diario['Fecha_Objetivo'] = balance_diario['Fecha'] + pd.Timedelta(days=1)
    balance_diario['Dia_Semana_Objetivo'] = balance_diario['Fecha_Objetivo'].dt.dayofweek
    balance_diario['Es_Fin_Semana_Objetivo'] = (
        balance_diario['Dia_Semana_Objetivo'] >= 5
    ).astype(int)
    balance_diario['Mes_Objetivo'] = balance_diario['Fecha_Objetivo'].dt.month
    objetivos = balance_diario[['Estacion_Id', 'Fecha', 'Estado_Actual']].rename(
        columns={'Fecha': 'Fecha_Objetivo', 'Estado_Actual': 'Riesgo_T1'}
    )
    dataset_ml = balance_diario.merge(
        objetivos, on=['Estacion_Id', 'Fecha_Objetivo'], how='left', validate='one_to_one'
    )
    dataset_ml['Riesgo_T1'] = dataset_ml['Riesgo_T1'].astype('Int64')
    # Keep rows without a future label until all lag and rolling features are computed.
    return dataset_ml


def creacion_variables_memoria_historica(dataset_ml):
    """Calcula historia en días consecutivos y luego elimina filas sin historia u objetivo."""
    # Asegurar de que el dataset esté perfectamente ordenado
    dataset_ml = dataset_ml.sort_values(['Estacion_Id', 'Fecha']).copy()
    if dataset_ml.duplicated(['Estacion_Id', 'Fecha']).any():
        raise ValueError('P2 requiere una fila por estación y día.')
    diferencias = dataset_ml.groupby('Estacion_Id')['Fecha'].diff().dropna()
    if diferencias.ne(pd.Timedelta(days=1)).any():
        raise ValueError('Completa el calendario diario antes de calcular los rezagos de P2.')

    # Forecast at the close of T: today's flows and balance are already available.
    if not (np.isfinite(dataset_ml['dpcapacity']) & dataset_ml['dpcapacity'].gt(0)).all():
        raise ValueError('P2 requiere capacidad positiva y finita para normalizar los flujos.')
    for columna in ('Llegadas', 'Salidas'):
        dataset_ml[f'{columna}_Por_Capacidad'] = dataset_ml[columna] / dataset_ml['dpcapacity']

    columnas_balance = []
    for lag in (1, 2, 3, 7):
        nombre = f'Balance_Lag_{lag}'
        dataset_ml[nombre] = dataset_ml.groupby('Estacion_Id')['Balance_Normalizado'].shift(lag)
        columnas_balance.append(nombre)
    for ventana in (3, 7):
        dataset_ml[f'Media_Movil_{ventana}d_Balance'] = dataset_ml.groupby(
            'Estacion_Id'
        )['Balance_Normalizado'].transform(
            lambda x: x.rolling(window=ventana, min_periods=ventana).mean()
        )

    # Volumen de llegadas y salidas en la misma estación en días anteriores.
    columnas_lag = []
    for lag in (1, 2, 3, 7):
        for columna in ('Llegadas', 'Salidas'):
            nombre_lag = f'{columna}_Lag_{lag}'
            dataset_ml[nombre_lag] = dataset_ml.groupby('Estacion_Id')[columna].shift(lag)
            columnas_lag.append(nombre_lag)
            nombre_normalizado = f'{columna}_Por_Capacidad_Lag_{lag}'
            dataset_ml[nombre_normalizado] = dataset_ml.groupby('Estacion_Id')[
                f'{columna}_Por_Capacidad'
            ].shift(lag)
            columnas_lag.append(nombre_normalizado)

    # Promedios móviles de 7 días para las llegadas y las salidas.
    dataset_ml['Media_Movil_7d_Llegadas'] = dataset_ml.groupby('Estacion_Id')['Llegadas'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    dataset_ml['Media_Movil_7d_Salidas'] = dataset_ml.groupby('Estacion_Id')['Salidas'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    for columna in ('Llegadas', 'Salidas'):
        dataset_ml[f'Media_Movil_7d_{columna}_Por_Capacidad'] = dataset_ml.groupby(
            'Estacion_Id'
        )[f'{columna}_Por_Capacidad'].transform(
            lambda x: x.rolling(window=7, min_periods=7).mean()
        )

    #  Limpieza post-ingeniería
    # Al hacer shift(7), los primeros 7 días de cada estación quedarán con nulos en Lag_7.
    # Discard unknown future labels only after computing history on the full calendar.
    dataset_ml = dataset_ml.dropna(subset=[*columnas_balance, *columnas_lag, 'Riesgo_T1']).copy()
    dataset_ml['Riesgo_T1'] = dataset_ml['Riesgo_T1'].astype(int)

    print(
        dataset_ml[
            [
                'Fecha',
                'Estacion_Id',
                'Balance_Lag_7',
                'Llegadas_Lag_1',
                'Llegadas_Lag_2',
                'Llegadas_Lag_3',
                'Llegadas_Lag_7',
                'Salidas_Lag_1',
                'Salidas_Lag_2',
                'Salidas_Lag_3',
                'Salidas_Lag_7',
                'Media_Movil_7d_Llegadas',
                'Media_Movil_7d_Salidas',
                'Riesgo_T1',
            ]
        ].head(10)
    )
    return dataset_ml

# ---------------------------------------------------------------------------------------------
# P3: grupos de estaciones con patrones de uso similares
# ---------------------------------------------------------------------------------------------


def filtrar_viajes_validos(viajes: pd.DataFrame) -> pd.DataFrame:
    """Quita viajes sin estación o fecha para que puedan resumirse por día.

    Los timestamps del dataset están truncados al día, así que no permiten validar la duración
    ni detectar devoluciones inmediatas; solo se comprueba que la fecha de llegada no sea anterior.
    """
    cols = [STATION_COL, DESTINO_COL, FECHA_INICIO_COL, FECHA_FIN_COL]
    con_datos = viajes.dropna(subset=cols)
    validos = con_datos[con_datos[FECHA_FIN_COL] >= con_datos[FECHA_INICIO_COL]]
    logger.info(
        f"P3: viajes descartados por campos nulos o fechas invertidas: "
        f"{len(viajes) - len(validos):,}"
    )
    return validos


def filtrar_periodo(viajes: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Viajes que inician entre `inicio` y `fin` (fechas inclusivas).

    Los grupos se ajustan solo con el periodo de entrenamiento; el periodo posterior se usa
    únicamente para medir si los grupos se mantienen, sin influir en el ajuste.
    """
    t = viajes[FECHA_INICIO_COL]
    fin_excl = pd.Timestamp(fin) + pd.Timedelta(days=1)
    return viajes[(t >= pd.Timestamp(inicio)) & (t < fin_excl)]


def resumen_uso(viajes: pd.DataFrame) -> pd.DataFrame:
    """Peso del fin de semana, intensidad de uso y balance de cada estación.

    - `prop_fin_semana`: fracción de salidas en sábado y domingo (uso recreativo o de traslado).
    - `log_viajes_dia`: log del promedio de salidas por día desde que la estación tiene uso.
    - `balance`: (llegadas - salidas) / total; estaciones que tienden a llenarse o vaciarse,
      dato útil para dimensionar nuevas estaciones.
    """
    salidas = viajes.groupby(STATION_COL).size()
    llegadas = viajes.groupby(DESTINO_COL).size()
    fin_semana = viajes[viajes[FECHA_INICIO_COL].dt.dayofweek >= 5].groupby(STATION_COL).size()
    fechas = viajes[FECHA_INICIO_COL].dt.normalize()
    dias = (fechas.max() - fechas.groupby(viajes[STATION_COL]).min()).dt.days + 1

    ids = salidas.index
    llegadas = llegadas.reindex(ids, fill_value=0)
    resumen = pd.DataFrame(
        {
            "prop_fin_semana": fin_semana.reindex(ids, fill_value=0) / salidas,
            "log_viajes_dia": np.log1p(salidas / dias.reindex(ids)),
            "balance": (llegadas - salidas) / (llegadas + salidas),
        }
    )
    resumen.index.name = "estacion_id"
    return resumen


def perfil_estaciones(
    viajes: pd.DataFrame, inicio: str, fin: str, min_viajes: int = P3_MIN_VIAJES
) -> pd.DataFrame:
    """Tabla P3 por estación con uso de fin de semana, intensidad diaria y balance.

    Solo entran estaciones con al menos `min_viajes` salidas en el periodo.
    """
    df = filtrar_periodo(viajes, inicio, fin)
    perfil = resumen_uso(df)
    activas = df.groupby(STATION_COL).size()
    perfil = perfil[perfil.index.isin(activas[activas >= min_viajes].index)]
    return perfil.fillna(0).reset_index()


def columnas_perfil(perfil: pd.DataFrame, prefijo: str | None = None) -> list[str]:
    """Columnas de variables del perfil; con `prefijo` (p. ej. "sal") solo las de esa familia."""
    cols = [c for c in perfil.columns if c != "estacion_id"]
    return [c for c in cols if c.startswith(prefijo)] if prefijo else cols


def info_estaciones(nomenclatura: pd.DataFrame) -> pd.DataFrame:
    """Nombre, ubicación, capacidad y municipio de cada estación para ubicar los grupos.

    El municipio se toma del código del nombre (p. ej. "(GDL-001)"); permite resumir en qué
    zonas de la ciudad se concentra cada grupo.
    """
    info = nomenclatura.rename(
        columns={
            "id": "estacion_id",
            "name": "nombre",
            "latitude": "lat",
            "longitude": "lon",
            "dpcapacity": "capacidad",
        }
    )
    info["municipio"] = info["nombre"].str.extract(r"([A-Z]{3})-\d+")[0].fillna("OTRO")
    return info[["estacion_id", "nombre", "lat", "lon", "capacidad", "municipio"]]
