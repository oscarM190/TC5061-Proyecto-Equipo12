from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mi_bici_experiment.config import (
    DESTINO_COL,
    FECHA_FIN_COL,
    FECHA_INICIO_COL,
    INTERIM_DATA_DIR,
    P3_DEVOLUCION_S,
    P3_DURACION_MAX_S,
    P3_MIN_VIAJES,
    RAW_DATA_DIR,
    STATION_COL,
)

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


# ---------------------------------------------------------------------------------------------
# P3: grupos de estaciones con patrones de uso similares
# ---------------------------------------------------------------------------------------------


def filtrar_viajes_validos(viajes: pd.DataFrame) -> pd.DataFrame:
    """Quita viajes que no reflejan uso real de una estación.

    Se descartan registros con nulos, duraciones no positivas o mayores a 6 h y devoluciones en
    la misma estación en menos de 60 s (la bicicleta se retira y se regresa de inmediato).
    Si se conservaran, inflarían el uso de estaciones con fallas y deformarían su perfil.
    """
    cols = [STATION_COL, DESTINO_COL, FECHA_INICIO_COL, FECHA_FIN_COL]
    df = viajes.dropna(subset=cols)
    duracion = (df[FECHA_FIN_COL] - df[FECHA_INICIO_COL]).dt.total_seconds()
    devolucion = (df[STATION_COL] == df[DESTINO_COL]) & (duracion < P3_DEVOLUCION_S)
    validos = (duracion > 0) & (duracion <= P3_DURACION_MAX_S) & ~devolucion
    logger.info(f"P3: viajes descartados por duración o devolución: {(~validos).sum():,}")
    return df[validos]


def filtrar_periodo(viajes: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Viajes que inician entre `inicio` y `fin` (fechas inclusivas).

    Los grupos se ajustan solo con el periodo de entrenamiento; el periodo posterior se usa
    únicamente para medir si los grupos se mantienen, sin influir en el ajuste.
    """
    t = viajes[FECHA_INICIO_COL]
    fin_excl = pd.Timestamp(fin) + pd.Timedelta(days=1)
    return viajes[(t >= pd.Timestamp(inicio)) & (t < fin_excl)]


def perfil_horario(
    viajes: pd.DataFrame, col_estacion: str, col_fecha: str, prefijo: str
) -> pd.DataFrame:
    """Proporción de los viajes de días hábiles que ocurre en cada hora, por estación.

    Con salidas (origen e inicio) separa estaciones de origen matutino, típicas de zonas de
    vivienda, de las de destino matutino (oficinas y escuelas), que se invierten con las
    llegadas. Usar proporciones y no conteos evita que el volumen domine la forma del día.
    """
    habiles = viajes[viajes[col_fecha].dt.dayofweek < 5]
    conteo = pd.crosstab(habiles[col_estacion], habiles[col_fecha].dt.hour)
    conteo = conteo.reindex(columns=range(24), fill_value=0)
    perfil = conteo.div(conteo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    perfil.columns = [f"{prefijo}_h{h:02d}" for h in perfil.columns]
    perfil.index.name = "estacion_id"
    return perfil


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
    """Tabla de P3: una fila por estación con su perfil de uso en el periodo [inicio, fin].

    Une el perfil horario de salidas y de llegadas en días hábiles (48 columnas) con el resumen
    de uso. Solo entran estaciones con al menos `min_viajes` salidas en el periodo: con pocas
    observaciones el perfil horario es ruido y genera grupos artificiales.
    """
    df = filtrar_periodo(viajes, inicio, fin)
    salidas = perfil_horario(df, STATION_COL, FECHA_INICIO_COL, "sal")
    llegadas = perfil_horario(df, DESTINO_COL, FECHA_FIN_COL, "lle")
    perfil = salidas.join(llegadas, how="inner").join(resumen_uso(df), how="inner")
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


if __name__ == "__main__":
    app()
