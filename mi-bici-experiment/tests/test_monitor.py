from pathlib import Path

import pandas as pd
import pytest

from mi_bici_experiment.config import FECHA_INICIO_COL, OBJETIVO_COL, STATION_COL
from mi_bici_experiment.modeling import monitor
from mi_bici_experiment.modeling.monitor import (
    MonthlyDataUnavailable,
    save_p1_monthly_dataset,
    save_p1_october_dataset,
)


# Build October features from September history and exclude invalid stations.
def test_october_features_use_september_history(tmp_path):
    history_path = tmp_path / "history.parquet"
    input_path = tmp_path / "october.csv"
    stations_path = tmp_path / "stations.csv"
    output_path = tmp_path / "october_features.parquet"

    pd.DataFrame(
        {
            STATION_COL: [1] * 7,
            FECHA_INICIO_COL: pd.date_range("2026-09-24", periods=7),
        }
    ).to_parquet(history_path, index=False)
    october_dates = pd.date_range("2026-10-01", periods=8)
    pd.DataFrame(
        {
            "Origen_Id": [1] * 16 + [2] * 8,
            "Inicio_del_viaje": list(october_dates.repeat(2)) + list(october_dates),
        }
    ).to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "id": [1, 2],
            "longitude": [-103.3, float("nan")],
            "latitude": [20.7, 20.7],
            "dpcapacity": [20, 20],
        }
    ).to_csv(stations_path, index=False)

    save_p1_october_dataset(input_path, output_path, history_path, stations_path)
    result = pd.read_parquet(output_path)

    assert len(result) == 8
    assert result[STATION_COL].eq(1).all()
    assert result[OBJETIVO_COL].eq(2).all()
    assert result["month"].eq(10).all()
    assert result.loc[0, "trips_lag_1"] == 1
    assert result.loc[0, "trips_lag_7"] == 1
    assert result.loc[0, "trips_roll_mean_7"] == 1
    assert result.loc[7, "trips_lag_7"] == 2


# Write one station's trips for the given dates to a monthly CSV.
def _write_month_csv(path, dates, trips_per_day=1):
    pd.DataFrame(
        {
            "Origen_Id": [1] * (len(dates) * trips_per_day),
            "Inicio_del_viaje": list(dates.repeat(trips_per_day)),
        }
    ).to_csv(path, index=False)


# Cover every October day in both years using September trips for initial lags.
@pytest.mark.parametrize("year", [2025, 2026])
def test_monthly_dataset_covers_full_october_with_september_lags(tmp_path, year):
    stations_path = tmp_path / "stations.csv"
    pd.DataFrame(
        {"id": [1], "longitude": [-103.3], "latitude": [20.7], "dpcapacity": [20]}
    ).to_csv(stations_path, index=False)
    previous_csv = tmp_path / f"datos_abiertos_{year}_09.csv"
    monthly_csv = tmp_path / f"datos_abiertos_{year}_10.csv"
    _write_month_csv(previous_csv, pd.date_range(f"{year}-09-24", periods=7))
    _write_month_csv(monthly_csv, pd.date_range(f"{year}-10-01", periods=31), 2)

    output_path = tmp_path / "monthly.parquet"
    save_p1_monthly_dataset(monthly_csv, output_path, stations_path)
    result = pd.read_parquet(output_path)

    assert len(result) == 31
    assert result["month"].eq(10).all()
    assert result[OBJETIVO_COL].eq(2).all()
    assert result.loc[0, "trips_lag_1"] == 1
    assert result.loc[0, "trips_lag_7"] == 1
    assert result.loc[0, "trips_roll_mean_7"] == 1
    assert result.loc[7, "trips_lag_7"] == 2


# Reject a short or gapped month without creating an output dataset.
@pytest.mark.parametrize(
    "dates",
    [
        pd.date_range("2026-10-01", periods=8),
        pd.date_range("2026-10-01", periods=31).delete(14),
    ],
)
def test_monthly_dataset_rejects_incomplete_month(tmp_path, dates):
    monthly_csv = tmp_path / "datos_abiertos_2026_10.csv"
    _write_month_csv(monthly_csv, dates)
    output_path = tmp_path / "monthly.parquet"

    with pytest.raises(MonthlyDataUnavailable, match="does not cover all"):
        save_p1_monthly_dataset(monthly_csv, output_path)
    assert not output_path.exists()


# Require the previous month's CSV to calculate the first lag features.
def test_monthly_dataset_requires_previous_month_history(tmp_path):
    monthly_csv = tmp_path / "datos_abiertos_2026_10.csv"
    stations_path = tmp_path / "stations.csv"
    _write_month_csv(monthly_csv, pd.date_range("2026-10-01", periods=31))
    pd.DataFrame(
        {"id": [1], "longitude": [-103.3], "latitude": [20.7], "dpcapacity": [20]}
    ).to_csv(stations_path, index=False)

    with pytest.raises(MonthlyDataUnavailable, match="2026_09"):
        save_p1_monthly_dataset(monthly_csv, tmp_path / "monthly.parquet", stations_path)


# Create the current drift report and skip the prior-year report when its CSV is missing.
def test_missing_prior_year_skips_only_second_report(tmp_path, monkeypatch, capsys):
    monthly_csv = tmp_path / "datos_abiertos_2026_10.csv"
    requested = []
    reports = []

    # Return the current dataset and simulate a missing prior-year dataset.
    def save_month(path):
        requested.append(Path(path).name)
        if Path(path).name == monthly_csv.name:
            return tmp_path / "current.parquet"
        raise MonthlyDataUnavailable("prior-year CSV missing")

    # Record each requested report without writing an HTML file.
    def create_report(reference, current, output=monitor.REPORTS_DIR / "p1_data_drift.html"):
        reports.append((reference, current))
        return output

    monkeypatch.setattr(monitor, "save_p1_monthly_dataset", save_month)
    monkeypatch.setattr(monitor, "save_p1_training_dataset", lambda: tmp_path / "history.parquet")
    monkeypatch.setattr(monitor, "create_p1_drift_report", create_report)
    monkeypatch.setattr(monitor, "_print_drift_result", lambda *args: None)

    monitor.run_monitoring(monthly_csv)

    assert requested == ["datos_abiertos_2026_10.csv", "datos_abiertos_2025_10.csv"]
    assert len(reports) == 1
    assert "Prior-year drift test skipped: prior-year CSV missing" in capsys.readouterr().out


# Reassemble an MLflow-split CSV path containing spaces before monitoring.
def test_main_accepts_mlflow_monthly_csv_path_with_spaces(monkeypatch):
    monthly_csv = Path("folder with spaces") / "datos_abiertos_2026_09.csv"
    parts = str(monthly_csv).split(" ")
    parts[0] = '""' + parts[0]
    parts[-1] += '""'
    requested = []
    monkeypatch.setattr(monitor, "run_monitoring", requested.append)

    monitor.main(["--monthly-csv", *parts])

    assert requested == [monthly_csv]
