import argparse
from base64 import b64encode
from collections import Counter
import csv
from datetime import datetime
from itertools import pairwise
from math import isfinite, nan
from pathlib import Path
import re
from stat import S_IMODE
from tempfile import NamedTemporaryFile

from matplotlib.figure import Figure


METRICS = ("cpu_percent", "memory_used_mib", "memory_percent")
AXIS_LABELS = ("CPU (%)", "Memoria usada (MiB)", "Memoria (% del límite)")
# Leave a gap when the time between snapshots exceeds the two-second sampling cadence.
MAX_GAP_SECONDS = 3
SECTION_START = "<!-- mibici-docker-stats:start -->"
SECTION_END = "<!-- mibici-docker-stats:end -->"


def read_samples(path):
    required = {"timestamp_utc", "container_id", "container_name", *METRICS}
    containers = {}
    with path.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        if not rows.fieldnames:
            raise ValueError("El CSV no contiene muestras de Docker.")
        missing = required.difference(rows.fieldnames)
        if missing:
            raise ValueError(f"Faltan columnas del CSV: {', '.join(sorted(missing))}")
        for row in rows:
            try:
                timestamp = datetime.fromisoformat(row["timestamp_utc"])
                values = tuple(float(row[metric]) for metric in METRICS)
                container_id, name = row["container_id"], row["container_name"]
                if timestamp.utcoffset() is None or not container_id or not name:
                    raise ValueError("Falta la zona horaria o la identificación del contenedor")
                if not all(isfinite(value) and value >= 0 for value in values):
                    raise ValueError("Las métricas deben ser números finitos y no negativos")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Muestra inválida en la línea {rows.line_num}: {error}") from error
            _, samples = containers.setdefault(container_id, (name, {}))
            samples[timestamp] = values
    if not containers:
        raise ValueError("El CSV no contiene muestras de Docker.")
    return containers


def resource_figure(containers):
    timestamps = sorted({timestamp for _, samples in containers.values() for timestamp in samples})
    timeline = [timestamps[0]]
    for previous, current in pairwise(timestamps):
        gap = current - previous
        if gap.total_seconds() > MAX_GAP_SECONDS:
            timeline.append(previous + gap / 2)
        timeline.append(current)
    seconds = [(timestamp - timeline[0]).total_seconds() for timestamp in timeline]

    figure = Figure(figsize=(10, 8), layout="constrained")
    figure.suptitle("MiBici P1 | Uso de recursos por contenedor", fontsize=16)
    axes = figure.subplots(len(METRICS), 1, sharex=True)
    name_counts = Counter(name for name, _ in containers.values())
    for container_id in sorted(containers, key=lambda key: (containers[key][0], key)):
        name, samples = containers[container_id]
        label = f"{name} ({container_id[:12]})" if name_counts[name] > 1 else name
        for index, axis in enumerate(axes):
            values = [samples.get(timestamp, (nan, nan, nan))[index] for timestamp in timeline]
            axis.plot(seconds, values, label=label, linewidth=1.6, marker="o", markersize=3.5)

    for axis, label in zip(axes, AXIS_LABELS):
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Segundos desde la primera muestra")
    axes[0].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=min(3, len(containers)),
        borderaxespad=0, frameon=False,
    )
    return figure


def embed_graph(html_path, png_path):
    with html_path.open(encoding="utf-8", newline="") as source:
        html = source.read()
    body_end = re.search(r"</body>\s*</html>\s*\Z", html, re.IGNORECASE)
    if body_end is None:
        raise ValueError(f"El reporte HTML no tiene un cierre de documento válido: {html_path}")

    image = b64encode(png_path.read_bytes()).decode("ascii")
    section = (
        f"{SECTION_START}\n"
        '<section id="mibici-docker-stats" aria-labelledby="mibici-docker-stats-title" '
        'style="max-width: 1000px; margin: 2rem auto; padding: 0 1rem; font-family: sans-serif">\n'
        '  <h2 id="mibici-docker-stats-title">Uso de recursos de Docker</h2>\n'
        f'  <img src="data:image/png;base64,{image}" '
        'alt="Gráficas de CPU y memoria por contenedor a lo largo de la prueba" '
        'style="display: block; width: 100%; height: auto">\n'
        "</section>\n"
        f"{SECTION_END}"
    )
    before = html[:body_end.start()]
    if SECTION_START in before:
        start = before.index(SECTION_START)
        end = before.index(SECTION_END, start) + len(SECTION_END)
        before = before[:start] + section + before[end:]
    else:
        before += section + "\n"

    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=html_path.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(before + html[body_end.start():])
        temporary.chmod(S_IMODE(html_path.stat().st_mode))
        temporary.replace(html_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Genera las gráficas de CPU y memoria de Docker.")
    parser.add_argument("csv_path", type=Path, help="CSV de muestras generado durante la prueba")
    parser.add_argument("--html", type=Path, help="Reporte de Locust donde se incrustarán las gráficas")
    args = parser.parse_args()
    png_path = args.csv_path.with_suffix(".png")
    try:
        resource_figure(read_samples(args.csv_path)).savefig(png_path, dpi=160)
        print(f"Gráficas de Docker: {png_path}")
        if args.html:
            embed_graph(args.html, png_path)
            print(f"Gráficas incrustadas: {args.html}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"No se pudo generar el reporte de Docker: {error}\n")


if __name__ == "__main__":
    main()
