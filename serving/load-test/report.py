import argparse
from itertools import pairwise
from math import nan
from pathlib import Path

from matplotlib.figure import Figure

from run import aggregate_rows, request_totals


def make_report(prefix):
    total = request_totals(Path(f"{prefix}_final_stats.csv"))

    # Locust can write two samples in the same second. Keep the last snapshot.
    history = aggregate_rows(Path(f"{prefix}_stats_history.csv"))
    snapshots = {float(row["Timestamp"]): row for row in history}
    timestamps = sorted(snapshots)
    if len(timestamps) < 2:
        raise ValueError("Se necesitan al menos dos muestras de Locust para graficar la prueba.")

    seconds, p50, p95, requests_per_second, error_percent = [], [], [], [], []
    for start, end in pairwise(timestamps):
        previous, current = snapshots[start], snapshots[end]
        requests = int(current["Total Request Count"]) - int(previous["Total Request Count"])
        failures = int(current["Total Failure Count"]) - int(previous["Total Failure Count"])
        # An uninitialized Locust window reports 0 even when every request took at least 1 ms.
        has_latency = requests > 0 and (
            current["95%"] != "0" or float(current["Total Min Response Time"]) < 1
        )
        seconds.append(end - timestamps[0])
        p50.append(float(current["50%"]) if has_latency else nan)
        p95.append(float(current["95%"]) if has_latency else nan)
        requests_per_second.append(requests / (end - start))
        error_percent.append(100 * failures / requests if requests else nan)

    request_count = int(total["Request Count"])
    failure_count = int(total["Failure Count"])
    summary = (
        f"{request_count:,} peticiones  ·  {failure_count:,} fallos "
        f"({100 * failure_count / request_count:.2f} %)  ·  "
        f"{float(total['Requests/s']):.2f} peticiones/s  ·  p99 {total['99%']} ms"
    )

    peak_users = max(int(row["User Count"]) for row in history)
    figure = Figure(figsize=(10, 8), layout="constrained")
    figure.suptitle(f"MiBici P1 | POST /predict · {peak_users} usuarios\n{summary}", fontsize=15)
    latency, throughput, errors = figure.subplots(3, 1, sharex=True)
    latency.plot(seconds, p50, color="#176879", label="p50", marker=".", markersize=3)
    latency.plot(seconds, p95, color="#572d8a", label="p95", marker=".", markersize=3)
    latency.set_ylabel("Latencia (ms)")
    latency.legend(loc="upper right")
    throughput.plot(seconds, requests_per_second, color="#176879", marker=".", markersize=3)
    throughput.set_ylabel("Peticiones/s")
    errors.plot(seconds, error_percent, color="#b54823", marker=".", markersize=3, clip_on=False)
    errors.set_ylabel("Errores (%)")
    errors.set_xlabel("Segundos desde la primera muestra")
    errors.set_ylim(top=min(100, max(1, errors.get_ylim()[1])))
    errors.spines["bottom"].set_visible(False)
    for axis in (latency, throughput, errors):
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)

    output = Path(f"{prefix}.png")
    figure.savefig(output, dpi=160)
    print(summary)
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Genera las gráficas de una corrida de Locust.")
    parser.add_argument("prefix", type=Path, help="Prefijo de los CSV generados por Locust")
    args = parser.parse_args()
    try:
        make_report(args.prefix)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"No se pudo generar el reporte: {error}\n")


if __name__ == "__main__":
    main()
