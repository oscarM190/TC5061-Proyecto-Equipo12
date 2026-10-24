from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import ssl
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import polars as pl
import certifi

from mi_bici_experiment.config import (
    INTERIM_DATA_DIR,
    MI_BICI_CSVS,
    PROJ_ROOT,
    RAW_DATA_DIR,
)


class DatasetManager:
    # Trip CSV columns retained for the modeling dataset.
    TRIP_COLUMNS = (
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    )
    # Trip timestamp columns parsed and truncated to day boundaries.
    DATETIME_COLUMNS = ("Inicio_del_viaje", "Fin_del_viaje")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Calculate a file's SHA-256 digest.

        Args:
            path: File whose contents should be hashed.

        Returns:
            Lowercase SHA-256 hexadecimal digest.
        """
        # Hash fixed-size chunks so large data files do not need to fit in memory.
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _manifest_file_path(path: Path) -> str:
        """Format a file path for the manifest.

        Args:
            path: File whose path should be recorded.

        Returns:
            Project-relative path, or the filename if the file is outside the project.
        """
        # Prefer repository-relative paths, falling back to the filename for external files.
        try:
            return path.resolve().relative_to(PROJ_ROOT.resolve()).as_posix()
        except ValueError:
            return path.name

    def write_trips_manifest(
        self,
        input_path: Path = RAW_DATA_DIR,
        output_path: Path = INTERIM_DATA_DIR / "trips_10_2024_to_09_2026.parquet",
        manifest_path: Path = PROJ_ROOT / "data_manifest.json",
    ) -> Path:
        """Write a versioned manifest for the trips Parquet and its source CSVs.

        Args:
            input_path: Directory containing monthly trips and station CSVs.
            output_path: Trips Parquet file to describe.
            manifest_path: Destination path for the JSON manifest.

        Returns:
            Path to the written manifest.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        manifest_path = Path(manifest_path)

        # Hash every source CSV and the generated Parquet file.
        source_files = sorted(input_path.glob("datos_abiertos_*.csv"))
        nomenclature_path = input_path / "nomenclatura_2026_09.csv"
        files = [
            {
                "path": self._manifest_file_path(path),
                "role": "source",
                "sha256": self._sha256_file(path),
            }
            for path in [*source_files, nomenclature_path]
        ]
        files.append(
            {
                "path": self._manifest_file_path(output_path),
                "role": "output",
                "sha256": self._sha256_file(output_path),
            }
        )

        # Count rows from the Parquet and assemble the versioned manifest.
        record_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
        manifest = {
            "manifest_version": 1,
            "dataset": "trips_10_2024_to_09_2026",
            "record_count": record_count,
            "files": files,
        }

        # Write to a temporary file, then replace the manifest to avoid partial JSON.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(manifest, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")
            temporary_path.replace(manifest_path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

        return manifest_path

    def download_csvs(self, max_workers: int = 4) -> list[Path]:
        """Download configured MiBici CSVs concurrently into the raw data directory.

        Existing non-empty files are reused, so calling this method again only
        downloads files that are missing or incomplete.

        Args:
            max_workers: Maximum number of downloads to run concurrently.

        Returns:
            Paths to the downloaded or reused CSV files.
        """
        # Create the destination folder before starting the parallel downloads.
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ssl_context = ssl.create_default_context()
        # Add Mozilla's CA bundle for Python installations without system CAs.
        ssl_context.load_verify_locations(cafile=certifi.where())

        # Download each configured URL concurrently, returning paths in URL order.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            downloader = partial(self._download_csv, ssl_context=ssl_context)
            return list(executor.map(downloader, MI_BICI_CSVS))

    @staticmethod
    def _download_csv(url: str, ssl_context: ssl.SSLContext) -> Path:
        """Download one CSV to the raw data directory.

        Args:
            url: Source URL of the CSV file.
            ssl_context: SSL context used for the HTTPS request.

        Returns:
            Path to the downloaded or existing CSV file.
        """
        # Derive the local filename and reuse a complete existing download.
        filename = Path(urlsplit(url).path).name
        if not filename:
            raise ValueError(f"URL does not contain a filename: {url}")

        destination = RAW_DATA_DIR / filename
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        # Download to a temporary file and validate it before replacing the target.
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
            # Remove a partial file so a later retry can start cleanly.
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

        return destination

    def load_trips_dataset(self, input_path: Path = RAW_DATA_DIR) -> pl.LazyFrame:
        """Load and combine monthly trips as a lazy Polars frame.

        Args:
            input_path: Directory containing monthly trip CSV files.

        Returns:
            Lazy frame with selected trip columns and normalized timestamps.
        """
        # Validate the input directory and find its monthly trip files.
        input_path = Path(input_path)
        if not input_path.is_dir():
            raise NotADirectoryError(f"Trip data directory does not exist: {input_path}")

        csv_files = sorted(input_path.glob("datos_abiertos_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No monthly trip CSV files found in {input_path}")

        monthly_trips = []
        for csv_file in csv_files:
            # Inspect one timestamp to select the source file's date format.
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

            # Lazily select modeling fields and normalize timestamps to day boundaries.
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

        # Stack the monthly scans while allowing compatible schema differences.
        return pl.concat(monthly_trips, how="vertical_relaxed")

    def save_trips_dataset(
        self,
        input_path: Path = RAW_DATA_DIR,
        output_path: Path = INTERIM_DATA_DIR / "trips_10_2024_to_09_2026.parquet",
    ) -> Path:
        """Join station metadata and save trips with complete origin data.

        Args:
            input_path: Directory containing monthly trips and station CSVs.
            output_path: Destination path for the trips Parquet file.

        Returns:
            Path to the saved Parquet file.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        # Ensure the output folder exists and require the station catalog.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nomenclature_path = Path(input_path) / "nomenclatura_2026_09.csv"
        if not nomenclature_path.is_file():
            raise FileNotFoundError(f"Station nomenclature file not found: {nomenclature_path}")

        # Cast station IDs so they match the catalog key type.
        trips = self.load_trips_dataset(input_path).with_columns(
            [
                pl.col("Origen_Id").cast(pl.Int16),
                pl.col("Destino_Id").cast(pl.Int16),
            ]
        )
        # Read only the station fields used to validate and enrich each trip.
        stations = pl.scan_csv(nomenclature_path, encoding="utf8-lossy").select(
            pl.col("id").cast(pl.Int16).alias("Origen_Id"),
            pl.col("longitude").cast(pl.Float32, strict=False),
            pl.col("latitude").cast(pl.Float32, strict=False),
            pl.col("dpcapacity").cast(pl.Int8, strict=False),
        )
        # Keep trips with valid origin metadata, sort them, and write the Parquet.
        trips.join(stations, on="Origen_Id", how="inner").filter(
            pl.col("longitude").is_finite()
            & pl.col("latitude").is_finite()
            & pl.col("dpcapacity").is_not_null()
        ).sort("Inicio_del_viaje").rename(
            {column: column.lower() for column in self.TRIP_COLUMNS}
        ).sink_parquet(
            output_path
        )
        # Record the output row count and hashes after the Parquet is complete.
        self.write_trips_manifest(input_path, output_path)
        return output_path


if __name__ == "__main__":
    dataset = DatasetManager()
    dataset.download_csvs()
    dataset.save_trips_dataset()
