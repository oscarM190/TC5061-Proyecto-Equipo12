import csv
from math import isfinite
from pathlib import Path
from random import choice

from locust import HttpUser, between, events, task
from locust.stats import PERCENTILES_TO_REPORT, StatsCSV

SEED_CSV = Path(__file__).resolve().parents[1] / "seed" / "p1_muestra.csv"
FEATURE_TYPES = {
    "trips_lag_1": float,
    "trips_lag_7": float,
    "trips_roll_mean_7": float,
    "is_weekend": int,
    "origen_id": int,
    "dayofweek": int,
    "month": int,
}


def load_payloads():
    with SEED_CSV.open(encoding="utf-8", newline="") as sample:
        payloads = [
            {field: cast(row[field]) for field, cast in FEATURE_TYPES.items()}
            for row in csv.DictReader(sample)
        ]
    if not payloads:
        raise ValueError(f"La muestra de predicción está vacía: {SEED_CSV}")
    return payloads


@events.test_stop.add_listener
def save_final_stats(environment, **kwargs):
    options = environment.parsed_options
    if options is None or not options.csv_prefix:
        return

    # The periodic CSV writer can miss requests between its last sample and shutdown.
    stats = StatsCSV(environment, PERCENTILES_TO_REPORT)
    for suffix, write in (("stats", stats.requests_csv), ("failures", stats.failures_csv)):
        path = Path(f"{options.csv_prefix}_final_{suffix}.csv")
        with path.open("w", encoding="utf-8", newline="") as output:
            write(csv.writer(output))


class PredictionUser(HttpUser):
    wait_time = between(0.5, 1.5)
    payloads = load_payloads()

    @task
    def predict(self):
        with self.client.post(
            "/predict",
            json=choice(self.payloads),
            timeout=(2, 5),
            allow_redirects=False,
            catch_response=True,
        ) as response:
            if not response.status_code:
                # Locust records the original connection error or timeout.
                return
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                result = response.json()
            except ValueError:
                response.failure("Respuesta sin JSON válido")
                return
            if not isinstance(result, dict):
                response.failure("La respuesta debe ser un objeto JSON")
                return

            prediction = result.get("prediction")
            model = result.get("model")
            if (
                type(prediction) not in (int, float)
                or prediction < 0
                or (isinstance(prediction, float) and not isfinite(prediction))
                or not isinstance(model, str)
                or not model.strip()
            ):
                response.failure("Respuesta de predicción inválida")
