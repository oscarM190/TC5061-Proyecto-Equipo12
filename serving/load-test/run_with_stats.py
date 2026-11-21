import argparse
import csv
from datetime import UTC, datetime
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from time import monotonic

from run import HOST_RESULTS_DIR, RESULTS_DIR

ROOT = Path(__file__).resolve().parents[2]
SERVICES = ("api", "mlflow", "redis", "worker")
SAMPLE_INTERVAL = 2
SAMPLE_TIMEOUT = 5
CSV_HEADER = (
    "timestamp_utc", "service", "container_id", "container_name", "cpu_percent",
    "memory_used_mib", "memory_limit_mib", "memory_percent",
)
MEMORY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")


def memory_mib(value):
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(B|[KMGTPE]iB)\s*", value)
    if match is None:
        raise ValueError(f"Cantidad de memoria inválida: {value}")
    amount, unit = match.groups()
    return float(amount) * 1024 ** (MEMORY_UNITS.index(unit) - 2)


def sample_containers():
    active = subprocess.run(
        [
            "docker", "compose", "ps", "--status", "running", "--no-trunc",
            "--format", "{{.ID}} {{.Service}}", *SERVICES,
        ],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=SAMPLE_TIMEOUT,
    )
    services = dict(line.split() for line in active.stdout.splitlines() if line.strip())
    if not services:
        print("No hay contenedores de MiBici activos para muestrear.", file=sys.stderr)
        return []

    stats = subprocess.run(
        ["docker", "stats", "--no-stream", "--no-trunc", "--format", "{{json .}}", *services],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=SAMPLE_TIMEOUT,
    )
    timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if stats.returncode:
        print(f"docker stats devolvió {stats.returncode}. {stats.stderr.strip()}", file=sys.stderr)

    rows = []
    for line in stats.stdout.splitlines():
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
            container_id = sample["ID"]
            service = services[container_id]
            name = sample["Name"]
            used, limit = map(memory_mib, sample["MemUsage"].split("/"))
            cpu_percent = float(sample["CPUPerc"].removesuffix("%"))
            memory_percent = float(sample["MemPerc"].removesuffix("%"))
            metrics = (cpu_percent, used, limit, memory_percent)
            if limit == 0 or not all(isfinite(value) and value >= 0 for value in metrics):
                raise ValueError("CPU o memoria sin una medición válida")
        except (KeyError, ValueError, TypeError, AttributeError) as error:
            print(f"Se omitió una fila de docker stats. {error}", file=sys.stderr)
            continue
        rows.append((timestamp, service, container_id, name, *metrics))
    return rows


def monitor_locust(locust, output):
    writer = csv.writer(output)
    writer.writerow(CSV_HEADER)
    output.flush()
    sampled_services = set()
    while locust.poll() is None:
        next_sample = monotonic() + SAMPLE_INTERVAL
        try:
            rows = sample_containers()
        except (subprocess.SubprocessError, ValueError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            print(f"Falló una muestra de Docker. {detail.strip()}", file=sys.stderr)
            rows = []

        if locust.poll() is not None:
            break
        writer.writerows(rows)
        output.flush()
        sampled_services.update(row[1] for row in rows)
        try:
            locust.wait(timeout=max(0, next_sample - monotonic()))
        except subprocess.TimeoutExpired:
            pass
    return sampled_services


def stop_locust(locust, name):
    try:
        result = subprocess.run(
            ["docker", "stop", "--timeout", "2", name],
            cwd=ROOT, capture_output=True, text=True, check=False, timeout=8,
        )
        if result.returncode and "no such container" not in result.stderr.lower():
            print(f"No se pudo detener {name}. {result.stderr.strip()}", file=sys.stderr)
        locust.wait(timeout=5)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Falló el cierre de Locust. {error}", file=sys.stderr)
        locust.kill()


def generate_docker_report(prefix, html_exists):
    command = [
        "docker", "compose", "run", "--rm", "--no-deps", "-T", "--entrypoint", "python",
        "locust", "/load-test/report_docker_stats.py", f"{prefix}_docker_stats.csv",
    ]
    if html_exists:
        command.extend(("--html", f"{prefix}.html"))
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    except OSError as error:
        print(f"No se pudo generar el reporte de Docker. {error}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta Locust en Docker Compose y guarda CPU y memoria por contenedor.",
        usage="%(prog)s [-- OPCIONES_DE_LOCUST]",
    )
    parser.add_argument("locust_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    locust_args = parser.parse_args().locust_args
    if locust_args[:1] == ["--"]:
        locust_args.pop(0)
    elif locust_args:
        parser.error("Separa las opciones de Locust con --.")
    if any(arg.split("=", 1)[0] in {"--csv", "--html"} for arg in locust_args):
        parser.error("--csv y --html se asignan automáticamente a la carpeta de la corrida.")
    if shutil.which("docker") is None:
        parser.exit(1, "Docker no está disponible en PATH. Ejecuta el lanzador donde usas Docker Compose.\n")

    try:
        if {"-h", "--help", "-V", "--version"}.intersection(locust_args):
            return subprocess.run(
                ["docker", "compose", "run", "--rm", "--no-deps", "--build", "-T", "locust", *locust_args],
                cwd=ROOT, check=False,
            ).returncode

        startup = subprocess.run(["docker", "compose", "up", "--build", "--wait", "api"], cwd=ROOT, check=False)
        if startup.returncode:
            return startup.returncode

        run_id = datetime.now(UTC).strftime("p1-%Y%m%d-%H%M%S-%fZ")
        results = ROOT / HOST_RESULTS_DIR / run_id
        results.mkdir(parents=True)
        prefix = PurePosixPath(RESULTS_DIR.as_posix()) / run_id / "p1"
        name = f"mibici-locust-{run_id}"
        stats_path = results / "p1_docker_stats.csv"
        command = [
            "docker", "compose", "run", "--rm", "--no-deps", "-T", "--name", name,
            "locust", "--csv", str(prefix), *locust_args,
        ]
        with stats_path.open("x", encoding="utf-8", newline="") as output:
            print(f"Muestras de Docker: {stats_path.relative_to(ROOT)}", flush=True)
            with subprocess.Popen(command, cwd=ROOT) as locust:
                try:
                    sampled_services = monitor_locust(locust, output)
                except (KeyboardInterrupt, OSError):
                    stop_locust(locust, name)
                    raise

        report_status = 0
        if sampled_services:
            report_status = generate_docker_report(prefix, (results / "p1.html").is_file())
        if "api" not in sampled_services:
            print("El CSV no contiene muestras válidas de la API.", file=sys.stderr)
        return locust.returncode or int("api" not in sampled_services) or report_status
    except KeyboardInterrupt:
        print("Prueba interrumpida.", file=sys.stderr)
        return 130
    except OSError as error:
        print(f"No se pudo ejecutar la prueba con Docker. {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
