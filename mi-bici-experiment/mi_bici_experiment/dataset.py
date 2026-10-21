import shutil
import ssl
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import polars as pl

from mi_bici_experiment.config import INTERIM_DATA_DIR, MI_BICI_CSVS, RAW_DATA_DIR


class Dataset:
    TRIP_COLUMNS = (
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    )
    DATETIME_COLUMNS = ("Inicio_del_viaje", "Fin_del_viaje")

    def download_csvs(self, max_workers: int = 4) -> list[Path]:
        """Download configured MiBici CSVs concurrently into the raw data directory.

        Existing non-empty files are reused, so calling this method again only
        downloads files that are missing or incomplete.
        """
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ssl_context = ssl.create_default_context()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            downloader = partial(self._download_csv, ssl_context=ssl_context)
            return list(executor.map(downloader, MI_BICI_CSVS))

    @staticmethod
    def _download_csv(url: str, ssl_context: ssl.SSLContext) -> Path:
        filename = Path(urlsplit(url).path).name
        if not filename:
            raise ValueError(f"URL does not contain a filename: {url}")

        destination = RAW_DATA_DIR / filename
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        temporary_path = None
        try:
            request = Request(url, headers={"User-Agent": "mi-bici-experiment/1.0"})
            with NamedTemporaryFile(
                dir=RAW_DATA_DIR,
                prefix=f".{filename}.",
                suffix=".part",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                with urlopen(request, timeout=60, context=ssl_context) as response:
                    shutil.copyfileobj(response, temporary_file, length=1024 * 1024)

            if temporary_path.stat().st_size == 0:
                raise ValueError(f"Downloaded an empty CSV from {url}")

            temporary_path.replace(destination)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

        return destination

    def load_trips_dataset(self, input_path: Path = RAW_DATA_DIR) -> pl.LazyFrame:
        """Lazily stack monthly trip CSVs, keeping only the modeling columns."""
        input_path = Path(input_path)
        if not input_path.is_dir():
            raise NotADirectoryError(f"Trip data directory does not exist: {input_path}")

        csv_files = sorted(input_path.glob("datos_abiertos_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No monthly trip CSV files found in {input_path}")

        monthly_trips = []
        for csv_file in csv_files:
            sample = pl.read_csv(
                csv_file,
                columns=[self.DATETIME_COLUMNS[0]],
                n_rows=1,
                encoding="utf8-lossy",
            )
            if sample.is_empty():
                raise ValueError(f"Trip CSV is empty: {csv_file}")

            sample_timestamp = sample.item(0, 0)
            if sample_timestamp is None:
                raise ValueError(f"Missing timestamp in the first row of {csv_file}")
            date_format = "%d/%m/%Y %H:%M" if "/" in sample_timestamp else "%Y-%m-%d %H:%M:%S"

            monthly_trips.append(
                pl.scan_csv(csv_file, encoding="utf8-lossy")
                .select(list(self.TRIP_COLUMNS))
                .with_columns(
                    [
                        pl.col(column)
                        .str.to_datetime(format=date_format, strict=True)
                        .dt.truncate("1d")
                        .alias(column)
                        for column in self.DATETIME_COLUMNS
                    ]
                )
            )

        return pl.concat(monthly_trips, how="vertical_relaxed")

    def save_trips_dataset(
        self,
        input_path: Path = RAW_DATA_DIR,
        output_path: Path = INTERIM_DATA_DIR / "trips.parquet",
    ) -> Path:
        """Join origin metadata and keep trips with complete station data."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nomenclature_path = Path(input_path) / "nomenclatura_2026_09.csv"
        if not nomenclature_path.is_file():
            raise FileNotFoundError(f"Station nomenclature file not found: {nomenclature_path}")

        trips = self.load_trips_dataset(input_path).with_columns(
            [
                pl.col("Origen_Id").cast(pl.Int16),
                pl.col("Destino_Id").cast(pl.Int16),
            ]
        )
        stations = pl.scan_csv(nomenclature_path, encoding="utf8-lossy").select(
            pl.col("id").cast(pl.Int16).alias("Origen_Id"),
            pl.col("longitude").cast(pl.Float32, strict=False),
            pl.col("latitude").cast(pl.Float32, strict=False),
            pl.col("dpcapacity").cast(pl.Int8, strict=False),
        )
        trips.join(stations, on="Origen_Id", how="inner").filter(
            pl.col("longitude").is_finite()
            & pl.col("latitude").is_finite()
            & pl.col("dpcapacity").is_not_null()
        ).sort("Inicio_del_viaje").rename(
            {column: column.lower() for column in self.TRIP_COLUMNS}
        ).sink_parquet(
            output_path
        )
        return output_path


if __name__ == "__main__":
    dataset = Dataset()
    dataset.download_csvs()
    dataset.save_trips_dataset()
