import csv
from datetime import datetime
import hashlib
import io
import json
from unittest.mock import Mock
from urllib.error import URLError

import polars as pl
import pytest

import mi_bici_experiment.dataset as dataset_module
from mi_bici_experiment.dataset import DatasetManager


# Provide a fresh dataset manager to each test that requests one.
@pytest.fixture
def manager():
    return DatasetManager()


# Write a CSV with the supplied columns and rows, then return its path.
def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# Write trip rows using the required trip columns and any extra columns.
def write_trips(path, rows, extra_columns=()):
    return write_csv(path, [*DatasetManager.TRIP_COLUMNS, *extra_columns], rows)


# Build a trip row with timestamps, station IDs, and optional extra fields.
def trip(start, end, origin=1, destination=2, **extra):
    return {
        "Inicio_del_viaje": start,
        "Fin_del_viaje": end,
        "Origen_Id": origin,
        "Destino_Id": destination,
        **extra,
    }


# [TEST 1] A nonexistent input directory raises NotADirectoryError.
def test_load_requires_existing_directory(manager, tmp_path):
    with pytest.raises(NotADirectoryError):
        manager.load_trips_dataset(tmp_path / "missing")


# [TEST 2] A directory without monthly trip CSVs raises FileNotFoundError.
def test_load_requires_matching_csv(manager, tmp_path):
    (tmp_path / "other.csv").write_text("anything\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        manager.load_trips_dataset(tmp_path)


# [TEST 3] Both supported timestamp formats are parsed and truncated to midnight.
@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("27/10/2024 14:30", "28/10/2024 01:45"),
        ("2024-10-27 14:30:00", "2024-10-28 01:45:00"),
    ],
)
def test_load_parses_and_truncates_both_dates(manager, tmp_path, start, end):
    write_trips(tmp_path / "datos_abiertos_2024_10.csv", [trip(start, end)])

    result = manager.load_trips_dataset(tmp_path).collect()

    assert result["Inicio_del_viaje"].to_list() == [datetime(2024, 10, 27)]
    assert result["Fin_del_viaje"].to_list() == [datetime(2024, 10, 28)]


# [TEST 4] Monthly files with different date formats are combined, retaining only trip columns.
def test_load_combines_months_with_different_formats_and_selects_columns(manager, tmp_path):
    write_trips(
        tmp_path / "datos_abiertos_2024_10.csv",
        [trip("27/10/2024 14:30", "27/10/2024 15:00", extra="ignored")],
        extra_columns=("extra",),
    )
    write_trips(
        tmp_path / "datos_abiertos_2024_11.csv",
        [trip("2024-11-01 14:30:00", "2024-11-02 00:01:00", origin=3)],
    )
    (tmp_path / "unrelated.csv").write_text("not,a,trip\n", encoding="utf-8")

    result = manager.load_trips_dataset(tmp_path).collect()

    assert result.columns == list(manager.TRIP_COLUMNS)
    assert result["Origen_Id"].to_list() == [1, 3]
    assert result["Inicio_del_viaje"].to_list() == [datetime(2024, 10, 27), datetime(2024, 11, 1)]
    assert result["Fin_del_viaje"].to_list() == [datetime(2024, 10, 27), datetime(2024, 11, 2)]


# [TEST 5] A trip CSV containing only a header raises ValueError.
def test_load_rejects_header_only_csv(manager, tmp_path):
    write_trips(tmp_path / "datos_abiertos_empty.csv", [])
    with pytest.raises(ValueError, match="empty"):
        manager.load_trips_dataset(tmp_path).collect()


# [TEST 6] A missing timestamp in the first trip raises ValueError.
def test_load_rejects_missing_first_timestamp(manager, tmp_path):
    write_trips(tmp_path / "datos_abiertos_missing.csv", [trip("", "2024-10-27 15:00:00")])
    with pytest.raises(ValueError, match="Missing timestamp"):
        manager.load_trips_dataset(tmp_path).collect()


# [TEST 7] Invalid start or end timestamps raise an error when the lazy frame is collected.
@pytest.mark.parametrize("bad_column", ["Inicio_del_viaje", "Fin_del_viaje"])
def test_load_rejects_malformed_timestamps(manager, tmp_path, bad_column):
    row = trip("2024-10-27 14:30:00", "2024-10-27 15:00:00")
    row[bad_column] = "2024-99-99 14:30:00"
    write_trips(tmp_path / "datos_abiertos_invalid.csv", [row])
    with pytest.raises(pl.exceptions.InvalidOperationError):
        manager.load_trips_dataset(tmp_path).collect()


# [TEST 8] Each required trip column must be present in the input CSV.
@pytest.mark.parametrize("missing_column", DatasetManager.TRIP_COLUMNS)
def test_load_rejects_missing_required_column(manager, tmp_path, missing_column):
    columns = [column for column in manager.TRIP_COLUMNS if column != missing_column]
    row = trip("2024-10-27 14:30:00", "2024-10-27 15:00:00")
    row.pop(missing_column)
    write_csv(tmp_path / "datos_abiertos_missing.csv", columns, [row])
    with pytest.raises((pl.exceptions.ColumnNotFoundError, pl.exceptions.SchemaError)):
        manager.load_trips_dataset(tmp_path).collect()


# [TEST 9] Saving without the station catalog raises FileNotFoundError and skips the manifest.
def test_save_requires_station_catalog(manager, tmp_path, monkeypatch):
    manifest = Mock()
    monkeypatch.setattr(manager, "write_trips_manifest", manifest)
    with pytest.raises(FileNotFoundError, match="Station nomenclature"):
        manager.save_trips_dataset(tmp_path, tmp_path / "output.parquet")
    manifest.assert_not_called()


# [TEST 10] Saved trips have valid origin metadata, expected types, and ascending start dates.
# Unknown origins and stations with invalid coordinates or capacity are removed.
def test_save_enriches_filters_sorts_and_types_trips(manager, tmp_path, monkeypatch):
    stations = [
        {"id": 1, "longitude": -103.3, "latitude": 20.7, "dpcapacity": 20},
        {"id": 2, "longitude": -103.4, "latitude": 20.6, "dpcapacity": 15},
        {"id": 3, "longitude": "invalid", "latitude": 20.7, "dpcapacity": 20},
        {"id": 4, "longitude": "", "latitude": 20.7, "dpcapacity": 20},
        {"id": 5, "longitude": "inf", "latitude": 20.7, "dpcapacity": 20},
        {"id": 6, "longitude": -103.3, "latitude": "invalid", "dpcapacity": 20},
        {"id": 7, "longitude": -103.3, "latitude": "", "dpcapacity": 20},
        {"id": 8, "longitude": -103.3, "latitude": "-inf", "dpcapacity": 20},
        {"id": 9, "longitude": -103.3, "latitude": 20.7, "dpcapacity": "invalid"},
        {"id": 10, "longitude": -103.3, "latitude": 20.7, "dpcapacity": ""},
        {"id": 11, "longitude": -103.3, "latitude": 20.7, "dpcapacity": 128},
    ]
    write_csv(tmp_path / "nomenclatura_2026_09.csv", ["id", "longitude", "latitude", "dpcapacity"], stations)
    rows = [trip("2024-10-29 14:30:00", "2024-10-29 15:00:00", origin=1),
            trip("2024-10-27 14:30:00", "2024-10-28 15:00:00", origin=2)]
    rows.extend(trip("2024-10-28 14:30:00", "2024-10-28 15:00:00", origin=i) for i in range(3, 13))
    write_trips(tmp_path / "datos_abiertos_2024_10.csv", rows)
    manifest = Mock()
    monkeypatch.setattr(manager, "write_trips_manifest", manifest)
    output = tmp_path / "nested" / "trips.parquet"

    assert manager.save_trips_dataset(tmp_path, output) == output
    result = pl.read_parquet(output)

    assert result.schema == {
        "inicio_del_viaje": pl.Datetime("us"),
        "fin_del_viaje": pl.Datetime("us"),
        "origen_id": pl.Int16,
        "destino_id": pl.Int16,
        "longitude": pl.Float32,
        "latitude": pl.Float32,
        "dpcapacity": pl.Int8,
    }
    assert result["origen_id"].to_list() == [2, 1]
    assert result["inicio_del_viaje"].to_list() == [datetime(2024, 10, 27), datetime(2024, 10, 29)]
    assert result["longitude"].to_list() == pytest.approx([-103.4, -103.3])
    assert result["latitude"].to_list() == pytest.approx([20.6, 20.7])
    assert result["dpcapacity"].to_list() == [15, 20]
    manifest.assert_called_once_with(tmp_path, output)


# [TEST 11] An existing nonempty download is reused without making a network request.
def test_download_reuses_existing_file(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "RAW_DATA_DIR", tmp_path)
    urlopen = Mock()
    monkeypatch.setattr(dataset_module, "urlopen", urlopen)
    destination = tmp_path / "trips.csv"
    destination.write_bytes(b"existing")

    assert manager._download_csv("https://example.com/trips.csv", None) == destination
    assert destination.read_bytes() == b"existing"
    urlopen.assert_not_called()


# [TEST 12] A successful download saves the response bytes and leaves no temporary file.
def test_download_saves_response(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "RAW_DATA_DIR", tmp_path)
    urlopen = Mock(return_value=io.BytesIO(b"id,value\n1,2\n"))
    monkeypatch.setattr(dataset_module, "urlopen", urlopen)
    context = object()

    destination = manager._download_csv("https://example.com/trips.csv?download=1", context)

    assert destination == tmp_path / "trips.csv"
    assert destination.read_bytes() == b"id,value\n1,2\n"
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://example.com/trips.csv?download=1"
    assert urlopen.call_args.kwargs == {"timeout": 60, "context": context}
    assert not list(tmp_path.glob("*.part"))


# [TEST 13] Empty responses and network errors leave no destination or partial download.
@pytest.mark.parametrize("failure", ["empty", "network"])
def test_download_cleans_up_failed_download(manager, tmp_path, monkeypatch, failure):
    monkeypatch.setattr(dataset_module, "RAW_DATA_DIR", tmp_path)
    if failure == "empty":
        urlopen = Mock(return_value=io.BytesIO(b""))
        error = ValueError
    else:
        urlopen = Mock(side_effect=URLError("connection failed"))
        error = URLError
    monkeypatch.setattr(dataset_module, "urlopen", urlopen)

    with pytest.raises(error):
        manager._download_csv("https://example.com/trips.csv", None)
    assert not (tmp_path / "trips.csv").exists()
    assert not list(tmp_path.glob("*.part"))


# [TEST 14] A URL without a filename raises ValueError without making a request.
def test_download_rejects_url_without_filename(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "RAW_DATA_DIR", tmp_path)
    urlopen = Mock()
    monkeypatch.setattr(dataset_module, "urlopen", urlopen)
    with pytest.raises(ValueError, match="filename"):
        manager._download_csv("https://example.com/", None)
    urlopen.assert_not_called()


# [TEST 15] The manifest records the version, row count, source and output paths, and hashes.
def test_write_manifest_records_sources_output_and_hashes(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "PROJ_ROOT", tmp_path / "project")
    first = tmp_path / "datos_abiertos_2024_10.csv"
    second = tmp_path / "datos_abiertos_2024_11.csv"
    catalog = tmp_path / "nomenclatura_2026_09.csv"
    for path, content in [(first, b"first"), (second, b"second"), (catalog, b"stations")]:
        path.write_bytes(content)
    (tmp_path / "unrelated.csv").write_bytes(b"ignored")
    output = tmp_path / "trips.parquet"
    pl.DataFrame({"value": [1, 2, 3]}).write_parquet(output)
    manifest_path = tmp_path / "manifests" / "manifest.json"

    assert manager.write_trips_manifest(tmp_path, output, manifest_path) == manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == 1
    assert manifest["dataset"] == "trips_10_2024_to_09_2026"
    assert manifest["record_count"] == 3
    assert manifest["files"] == [
        {"path": path.name, "role": role, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path, role in [(first, "source"), (second, "source"), (catalog, "source"), (output, "output")]
    ]


# [TEST 16] The file hashing helper returns the known SHA-256 digest of the input bytes.
def test_sha256_file_matches_known_digest(manager, tmp_path):
    path = tmp_path / "example.txt"
    path.write_bytes(b"abc")
    assert manager._sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# [TEST 17] Manifest paths are project-relative inside the project and filenames outside it.
def test_manifest_file_path_inside_and_outside_project(manager, tmp_path, monkeypatch):
    project = tmp_path / "project"
    monkeypatch.setattr(dataset_module, "PROJ_ROOT", project)
    assert manager._manifest_file_path(project / "data" / "trips.csv") == "data/trips.csv"
    assert manager._manifest_file_path(tmp_path / "outside" / "trips.csv") == "trips.csv"
