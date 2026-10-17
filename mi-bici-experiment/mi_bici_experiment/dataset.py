from pathlib import Path
import pandas as pd
import typer
from mi_bici_experiment.config import INTERIM_DATA_DIR, RAW_DATA_DIR, MI_BICI_CSVS

import argparse
from pathlib import Path
import shutil
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

app = typer.Typer()

def read_csv(csv_file, **read_csv_kwargs):
    if read_csv_kwargs.get("encoding") is not None:
        return pd.read_csv(csv_file, **read_csv_kwargs)
    options = {**read_csv_kwargs, "encoding": "utf-8-sig"}
    try:
        return pd.read_csv(csv_file, **options)
    except UnicodeDecodeError:
        options["encoding"] = "latin-1"
        return pd.read_csv(csv_file, **options)

def load_trips_dataset(input_path: Path = RAW_DATA_DIR, **read_csv_kwargs) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise NotADirectoryError(f"No existe la carpeta de datos: {input_path}")

    csv_files = sorted(
        csv_file for csv_file in input_path.glob("*.csv")
        if "nomenclatura" not in csv_file.name.lower()
    )
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos .csv en {input_path}")

    return pd.concat(
        (read_csv(csv_file) for csv_file in csv_files),
        ignore_index=True,
        sort=False,
    )


def create_ssl_context(ca_file=None):
    context = ssl.create_default_context()
    if ca_file is not None:
        context.load_verify_locations(cafile=str(ca_file))
    else:
        try:
            import certifi
        except ImportError:
            pass
        else:
            context.load_verify_locations(cafile=certifi.where())
    return context


def download(url, directory, overwrite=False, attempts=3, timeout=60, context=None):
    destination = directory / Path(urlsplit(url).path).name
    if not overwrite and destination.exists() and destination.stat().st_size > 0:
        return "omitido"
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": "MiBici-CSV-Downloader/1.0"})
            with urlopen(request, timeout=timeout, context=context) as response:
                if "html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("El servidor devolvió HTML en lugar de un CSV")
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
                size = partial.stat().st_size
                expected = response.headers.get("Content-Length")
                if size == 0 or (expected is not None and size != int(expected)):
                    raise ValueError("La descarga está vacía o incompleta")
            partial.replace(destination)
            return "descargado"
        except (OSError, URLError, ValueError) as error:
            partial.unlink(missing_ok=True)
            if attempt == attempts or (
                isinstance(error, HTTPError) and 400 <= error.code < 500
                and error.code not in (408, 429)
            ):
                raise
            time.sleep(2 ** (attempt - 1))

def download_csvs_from_mibici(input_path: Path):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=input_path,
        help="Carpeta de destino (por defecto: mi-bici-experiment/downloads)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Volver a descargar existentes")
    parser.add_argument("--ca-file", type=Path, help="Certificados CA adicionales en formato PEM")
    args = parser.parse_args()
    try:
        context = create_ssl_context(args.ca_file)
    except (OSError, ssl.SSLError) as error:
        parser.error(f"No se pudieron cargar los certificados: {error}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"descargado": 0, "omitido": 0, "fallido": 0}
    index = 0
    for url in MI_BICI_CSVS:
        name = Path(urlsplit(url).path).name
        try:
            destination = args.output_dir / name
            if destination.is_file() and not args.overwrite:
                status = "omitido"
            else:
                status = download(url, args.output_dir, args.overwrite, context=context)
            counts[status] += 1
            print(f"[{index}/{MI_BICI_CSVS.count}] {status}: {name}", flush=True)
        except (OSError, URLError, ValueError) as error:
            counts["fallido"] += 1
            print(f"[{index}/{MI_BICI_CSVS.count}] fallido: {name}: {error}", file=sys.stderr, flush=True)
    print(f"Resultado: {counts}. Destino: {args.output_dir.resolve()}")
    return 1 if counts["fallido"] else 0


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR, 
):
    download_csvs_from_mibici(input_path)
    trips_df = load_trips_dataset()
    trips_df["Inicio_del_viaje"] = pd.to_datetime(
                trips_df["Inicio_del_viaje"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    trips_df["Fin_del_viaje"] = pd.to_datetime(
                    trips_df["Fin_del_viaje"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    
    nomenclatura_df = read_csv(RAW_DATA_DIR.joinpath("nomenclatura_2026_09.csv"))
    nomenclatura_df.to_parquet(RAW_DATA_DIR.joinpath("nomenclatura.parquet"))
    trips_df.to_parquet(RAW_DATA_DIR.joinpath("trips.parquet"))

if __name__ == "__main__":
    app()
