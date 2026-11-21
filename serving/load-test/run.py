import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import sys

RESULTS_DIR = Path("/results")
HOST_RESULTS_DIR = Path("serving/load-test/results")


def workspace_path(path):
    if path.is_relative_to(RESULTS_DIR):
        return HOST_RESULTS_DIR / path.relative_to(RESULTS_DIR)
    return path


def aggregate_rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        return [row for row in csv.DictReader(source) if row["Name"] == "Aggregated"]


def request_totals(path):
    totals = aggregate_rows(path)
    if not totals or int(totals[0]["Request Count"]) == 0:
        raise ValueError("Locust no registró peticiones. Revisa la salida de la prueba.")
    return totals[0]


def main():
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%fZ")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--csv", type=Path, default=RESULTS_DIR / f"p1-{timestamp}" / "p1")
    parser.add_argument("--html", type=Path)
    report_options, load_options = parser.parse_known_args()
    locust_command = [sys.executable, "-m", "locust"]
    if {"-h", "--help", "-V", "--version"}.intersection(load_options):
        return subprocess.run([*locust_command, *load_options], check=False).returncode

    prefix = report_options.csv
    html = report_options.html or Path(f"{prefix}.html")
    history = Path(f"{prefix}_stats_history.csv")
    final_stats = Path(f"{prefix}_final_stats.csv")
    final_failures = Path(f"{prefix}_final_failures.csv")
    artifacts = (html, final_stats, final_failures, history)
    previous_writes = {path: path.stat().st_mtime_ns for path in artifacts if path.is_file()}

    locust_status = subprocess.run(
        [
            *locust_command,
            "--csv", str(prefix),
            "--html", str(html),
            "--csv-full-history",
            "--only-summary",
            *load_options,
        ],
        check=False,
    ).returncode

    # Custom prefixes can be reused. Report only files written by this run.
    written = {
        path
        for path in artifacts
        if path.is_file() and path.stat().st_mtime_ns != previous_writes.get(path)
    }
    if written:
        print(f"Carpeta de resultados: {workspace_path(prefix.parent)}")
    for artifact in artifacts:
        if artifact in written:
            print(f"Reporte: {workspace_path(artifact)}")

    if locust_status:
        return locust_status
    if not {html, history, final_stats} <= written:
        print("Locust no generó el HTML y los CSV necesarios para el reporte.", file=sys.stderr)
        return 1
    try:
        request_totals(final_stats)
    except (OSError, ValueError, KeyError) as error:
        print(f"No se pudo validar el resultado de Locust: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
