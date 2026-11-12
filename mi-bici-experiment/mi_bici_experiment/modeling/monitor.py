"""Export P1 datasets and compare their feature distributions."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mi_bici_experiment.config import (
    FECHA_INICIO_COL,
    LAGS,
    NOMENCLATURA_CSV,
    OBJETIVO_COL,
    PM_DIAS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    STATION_COL,
    TRIPS_PARQUET,
)
from mi_bici_experiment.features import (
    agregar_variables,
    construir_viajes_diarios,
    limpiar_viajes,
)


class MonthlyDataUnavailable(ValueError):
    """A monthly CSV or its required lag history is unavailable."""


def _period_from_csv(path: Path) -> pd.Period:
    # Use the filename period to validate the dates in the monthly export.
    match = re.fullmatch(r"datos_abiertos_(\d{4})_(0[1-9]|1[0-2])\.csv", Path(path).name)
    if match is None:
        raise ValueError(f"Expected a monthly CSV named datos_abiertos_YYYY_MM.csv: {path}")
    return pd.Period(year=int(match[1]), month=int(match[2]), freq="M")


def _csv_for_period(directory: Path, period: pd.Period) -> Path:
    return directory / f"datos_abiertos_{period.year}_{period.month:02d}.csv"


def _valid_station_ids(stations_path: Path) -> pd.Series:
    stations = pd.read_csv(
        stations_path,
        usecols=["id", "longitude", "latitude", "dpcapacity"],
        encoding_errors="replace",
    )
    for column in ("id", "longitude", "latitude", "dpcapacity"):
        stations[column] = pd.to_numeric(stations[column], errors="coerce")
    # Keep stations with finite coordinates and a present numeric capacity.
    return stations.loc[
        np.isfinite(stations["longitude"])
        & np.isfinite(stations["latitude"])
        & stations["dpcapacity"].notna(),
        "id",
    ]


def _read_trips_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise MonthlyDataUnavailable(f"Monthly trip CSV not found: {path}")
    trips = pd.read_csv(
        path,
        usecols=["Origen_Id", "Inicio_del_viaje"],
        encoding_errors="replace",
    ).rename(columns={"Origen_Id": STATION_COL, "Inicio_del_viaje": FECHA_INICIO_COL})
    trips[STATION_COL] = pd.to_numeric(trips[STATION_COL], errors="coerce")
    # Parse older day-first exports before the shared trip cleaning step.
    sample_dates = trips[FECHA_INICIO_COL].dropna()
    if not sample_dates.empty and "/" in str(sample_dates.iloc[0]):
        trips[FECHA_INICIO_COL] = pd.to_datetime(
            trips[FECHA_INICIO_COL], errors="coerce", dayfirst=True
        )
    return limpiar_viajes(trips, STATION_COL, FECHA_INICIO_COL)


def _keep_valid_stations(trips: pd.DataFrame, station_ids: pd.Series) -> pd.DataFrame:
    # Trips with unknown origins cannot receive valid station metadata.
    valid = trips[trips[STATION_COL].isin(station_ids)].copy()
    if not valid.empty:
        valid[STATION_COL] = valid[STATION_COL].astype("int16")
    return valid


def _check_history(history: pd.DataFrame, first_day: pd.Timestamp, source: Path) -> None:
    history_days = max(*LAGS, PM_DIAS)
    # Require every prior day needed by the lag and rolling features.
    expected_days = pd.date_range(end=first_day - pd.Timedelta(days=1), periods=history_days)
    actual_days = pd.DatetimeIndex(history[FECHA_INICIO_COL].dt.normalize().unique())
    if not expected_days.isin(actual_days).all():
        raise MonthlyDataUnavailable(f"{source} lacks the {history_days} days before {first_day.date()}")


def _save_featured_month(
    trips: pd.DataFrame,
    history: pd.DataFrame,
    period: pd.Period,
    output_path: Path,
) -> Path:
    from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionModel

    # Build lag features with prior history before keeping the requested month.
    combined = pd.concat([history, trips], ignore_index=True)
    daily = construir_viajes_diarios(combined, STATION_COL, FECHA_INICIO_COL)
    featured = agregar_variables(daily, STATION_COL)
    monthly = featured[featured["date"].dt.to_period("M").eq(period)]
    if monthly.empty:
        raise MonthlyDataUnavailable(f"No P1 feature rows for {period}")
    # Export features in the same order expected by the P1 model.
    model = RegressionModel()
    monthly = monthly[model.todas_var + [OBJETIVO_COL]]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(output_path, index=False)
    return output_path


def save_p1_training_dataset(
    output_path: Path = PROCESSED_DATA_DIR / "p1_train.parquet",
) -> Path:
    """Save P1 train, validation, and test rows with the model columns."""
    from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionModel

    model = RegressionModel()
    model.preparar_datos()
    # Combine the chronological splits using the model's feature columns.
    all_data = pd.concat(
        [model.entrenamiento, model.validacion, model.prueba], ignore_index=True
    )[model.todas_var + [OBJETIVO_COL]]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_data.to_parquet(output_path, index=False)
    return output_path


def save_p1_october_dataset(
    input_path: Path = RAW_DATA_DIR / "datos_abiertos_2026_10.csv",
    output_path: Path = PROCESSED_DATA_DIR / "p1_2026_10.parquet",
    history_path: Path = TRIPS_PARQUET,
    stations_path: Path = NOMENCLATURA_CSV,
) -> Path:
    """Keep the existing October export interface for callers and tests."""
    input_path = Path(input_path)
    history_path = Path(history_path)
    trips = _keep_valid_stations(_read_trips_csv(input_path), _valid_station_ids(stations_path))
    if trips.empty:
        raise ValueError(f"No valid P1 trips found in {input_path}")
    first_day = trips[FECHA_INICIO_COL].min().normalize()
    history_days = max(*LAGS, PM_DIAS)
    # Read only the preceding days needed to seed October's lag features.
    history = pd.read_parquet(
        history_path,
        columns=[STATION_COL, FECHA_INICIO_COL],
        filters=[
            (FECHA_INICIO_COL, ">=", first_day - pd.Timedelta(days=history_days)),
            (FECHA_INICIO_COL, "<", first_day),
        ],
    )
    _check_history(history, first_day, history_path)
    return _save_featured_month(trips, history, pd.Period(first_day, freq="M"), output_path)


def save_p1_monthly_dataset(
    input_path: Path,
    output_path: Path | None = None,
    stations_path: Path = NOMENCLATURA_CSV,
) -> Path:
    """Create a complete month's P1 features from its CSV and the preceding CSV."""
    input_path = Path(input_path)
    period = _period_from_csv(input_path)
    trips = _read_trips_csv(input_path)
    first_day = period.start_time
    last_day = (period + 1).start_time - pd.Timedelta(days=1)
    dates = trips[FECHA_INICIO_COL].dt.normalize()
    # Check source coverage before filtering trips by valid station IDs.
    if (
        trips.empty
        or not dates.dt.to_period("M").eq(period).all()
        or dates.min() != first_day
        or dates.max() != last_day
        or dates.nunique() != period.days_in_month
    ):
        raise MonthlyDataUnavailable(f"{input_path} does not cover all of {period}")

    station_ids = _valid_station_ids(stations_path)
    trips = _keep_valid_stations(trips, station_ids)
    if trips.empty:
        raise MonthlyDataUnavailable(f"No valid P1 trips found in {input_path}")

    # Limit the previous month's trips to the required lookback window.
    history_path = _csv_for_period(input_path.parent, period - 1)
    history = _keep_valid_stations(_read_trips_csv(history_path), station_ids)
    history_days = max(*LAGS, PM_DIAS)
    history = history[
        (history[FECHA_INICIO_COL] >= first_day - pd.Timedelta(days=history_days))
        & (history[FECHA_INICIO_COL] < first_day)
    ]
    _check_history(history, first_day, history_path)

    if output_path is None:
        output_path = PROCESSED_DATA_DIR / f"p1_{period.year}_{period.month:02d}.parquet"
    return _save_featured_month(trips, history, period, output_path)


def create_p1_drift_report(
    reference_path: Path = PROCESSED_DATA_DIR / "p1_train.parquet",
    current_path: Path = PROCESSED_DATA_DIR / "p1_2026_10.parquet",
    output_path: Path = REPORTS_DIR / "p1_data_drift.html",
) -> Path:
    """Save HTML and JSON drift results for current P1 features against reference data."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionModel

    model = RegressionModel()
    features = model.todas_var
    reference = pd.read_parquet(reference_path, columns=features)
    current = pd.read_parquet(current_path, columns=features)
    if reference.empty or current.empty:
        raise ValueError("Both P1 datasets must have rows for drift analysis")

    # Use the model's feature types for numerical and categorical drift checks.
    schema = DataDefinition(
        numerical_columns=model.num_var,
        categorical_columns=model.cat_var,
    )
    reference_data = Dataset.from_pandas(reference, data_definition=schema)
    current_data = Dataset.from_pandas(current, data_definition=schema)
    snapshot = Report([DataDriftPreset(columns=features)], include_tests=True).run(
        current_data, reference_data
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the readable report and its machine-readable metrics together.
    snapshot.save_html(str(output_path))
    snapshot.save_json(str(output_path.with_suffix(".json")))
    return output_path


def _print_drift_result(label: str, report_path: Path) -> None:
    result = json.loads(report_path.with_suffix(".json").read_text(encoding="utf-8"))
    # Find the metric that counts features flagged for drift.
    summary = next(
        metric
        for metric in result["metrics"]
        if metric["config"]["type"].endswith("DriftedColumnsCount")
    )
    print(
        f"{label}: {int(summary['value']['count'])}/{len(summary['config']['columns'])} "
        f"features drifted | HTML: {report_path} | JSON: {report_path.with_suffix('.json')}"
    )


def run_monitoring(monthly_csv: Path) -> None:
    """Prepare monthly data and run whole-history and prior-year drift checks."""
    monthly_csv = Path(monthly_csv)
    period = _period_from_csv(monthly_csv)
    # Compare the current month with the combined training data first.
    monthly_path = save_p1_monthly_dataset(monthly_csv)
    training_path = save_p1_training_dataset()
    report_path = create_p1_drift_report(training_path, monthly_path)
    print(f"P1 train/validation/test dataset saved to {training_path}")
    print(f"P1 {period} dataset saved to {monthly_path}")
    _print_drift_result("Whole history vs current month", report_path)

    # Missing prior-year data skips only this seasonal comparison.
    previous_period = period - 12
    previous_csv = _csv_for_period(monthly_csv.parent, previous_period)
    try:
        previous_path = save_p1_monthly_dataset(previous_csv)
    except MonthlyDataUnavailable as error:
        print(f"Prior-year drift test skipped: {error}")
        return

    previous_label = f"{previous_period.year}_{previous_period.month:02d}"
    current_label = f"{period.year}_{period.month:02d}"
    seasonal_report = REPORTS_DIR / f"p1_data_drift_{previous_label}_vs_{current_label}.html"
    create_p1_drift_report(previous_path, monthly_path, seasonal_report)
    print(f"P1 {previous_period} dataset saved to {previous_path}")
    _print_drift_result("Same month last year vs current month", seasonal_report)


def main(argv: list[str] | None = None) -> None:
    """Run with ``python -m mi_bici_experiment.modeling.monitor``."""
    parser = argparse.ArgumentParser(description="Create two P1 monthly data drift reports")
    parser.add_argument(
        "--monthly-csv",
        type=Path,
        nargs="+",
        default=[RAW_DATA_DIR / "datos_abiertos_2026_10.csv"],
        help="New monthly trip CSV (default: October 2026)",
    )
    args = parser.parse_args(argv)
    # MLflow's Windows cmd launcher can split a quoted path containing spaces.
    monthly_csv = Path(" ".join(map(str, args.monthly_csv)).strip('"'))
    run_monitoring(monthly_csv)


if __name__ == "__main__":
    main()
