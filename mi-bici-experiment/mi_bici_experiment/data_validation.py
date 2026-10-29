"""Validate the interim trips Parquet with Deepchecks.

Run this module in the separate environment described in requirements-deepchecks.txt.
The full file is scanned for schema and null counts; Deepchecks uses bounded,
uniform samples from the two chronological periods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = PROJECT_ROOT / "data/interim/trips_10_2024_to_09_2026.parquet"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/deepchecks"
DATE_COLUMN = "inicio_del_viaje"
COLUMNS = (
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
    "longitude",
    "latitude",
    "dpcapacity",
)
DRIFT_COLUMNS = ("origen_id", "destino_id", "longitude", "latitude", "dpcapacity")


def check_schema(schema: pa.Schema):
    """Check that a Parquet schema contains the required trip columns and types.

    Extra columns are allowed. The required timestamps must use a timestamp type,
    station IDs and capacity must use integer types, and coordinates must use
    floating-point types. This checks broad Arrow type families rather than
    requiring particular widths or timestamp units.

    Args:
        schema: Arrow schema read from the Parquet file.

    Raises:
        ValueError: If any required column is missing or has an incompatible type.
    """
    missing = sorted(set(COLUMNS) - set(schema.names))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    expected = {
        "inicio_del_viaje": pa.types.is_timestamp,
        "fin_del_viaje": pa.types.is_timestamp,
        "origen_id": pa.types.is_integer,
        "destino_id": pa.types.is_integer,
        "longitude": pa.types.is_floating,
        "latitude": pa.types.is_floating,
        "dpcapacity": pa.types.is_integer,
    }
    wrong = [
        f"{name}: {schema.field(name).type}"
        for name, accepts in expected.items()
        if not accepts(schema.field(name).type)
    ]
    if wrong:
        raise ValueError(f"Unexpected Parquet types: {', '.join(wrong)}")


def _keep_smallest(
    current: pd.DataFrame,
    scores: np.ndarray,
    chunk: pd.DataFrame,
    new_scores: np.ndarray,
    limit: int,
):
    """Retain the rows with the lowest random priorities up to ``limit``.

    This helper merges an existing sample with a new batch, then discards rows
    whose random priorities are too high. Keeping the smallest priorities gives
    a bounded, uniform sample across batches without storing every row.

    Args:
        current: Previously retained rows, with one priority per row in ``scores``.
        scores: Random priorities corresponding to ``current``.
        chunk: Newly scanned rows to consider for the sample.
        new_scores: Random priorities corresponding to the rows in ``chunk``.
        limit: Maximum number of rows to retain; must be positive.

    Returns:
        A pair containing the retained rows as a DataFrame and their priorities
        as a NumPy array. The two outputs remain aligned by position.
    """
    if len(new_scores) > limit:
        chosen = np.argpartition(new_scores, limit - 1)[:limit]
        chunk = chunk.iloc[chosen]
        new_scores = new_scores[chosen]
    combined = pd.concat((current, chunk), ignore_index=True)
    combined_scores = np.concatenate((scores, new_scores))
    if len(combined) > limit:
        chosen = np.argpartition(combined_scores, limit - 1)[:limit]
        return combined.iloc[chosen].reset_index(drop=True), combined_scores[chosen]
    return combined, combined_scores


def scan_and_sample(
    parquet_path: Path, cutoff: pd.Timestamp, sample_size: int, batch_size: int = 65_536
):
    """Audit a trips Parquet file and sample rows from both sides of a cutoff.

    The file is read in Arrow batches so the full dataset does not need to fit
    in memory. Every required column is checked for nulls, and rows are counted
    for the full file. Non-null start dates before ``cutoff`` belong to the
    ``train`` period; dates on or after it belong to ``test``. Rows with a null
    start date contribute to the total and null counts but to neither period.
    Each period is sampled independently, with at most ``sample_size`` rows.
    A fixed random seed makes the selected samples reproducible.

    Args:
        parquet_path: Parquet file containing the trip records.
        cutoff: Timestamp that separates the train and test periods.
        sample_size: Maximum number of rows to sample from each period. Use a
            positive value.
        batch_size: Number of Parquet rows to read per batch.

    Returns:
        A tuple of ``(audit, train_sample, test_sample)``. ``audit`` contains
        total and per-period row counts, required-column null counts, the
        cutoff date, and actual sample sizes. The two samples are pandas
        DataFrames with the required trip columns.

    Raises:
        ValueError: If the file schema is invalid or either period has no rows.
    """
    source = pq.ParquetFile(parquet_path)
    check_schema(source.schema_arrow)
    rng = np.random.default_rng(42)
    samples = {part: pd.DataFrame(columns=COLUMNS) for part in ("train", "test")}
    priorities = {part: np.empty(0) for part in samples}
    counts = {part: 0 for part in samples}
    null_counts = {column: 0 for column in COLUMNS}

    for batch in source.iter_batches(batch_size=batch_size, columns=list(COLUMNS)):
        for column in COLUMNS:
            null_counts[column] += batch.column(column).null_count
        frame = batch.to_pandas()
        train_mask = frame[DATE_COLUMN] < cutoff
        for part, mask in (
            ("train", train_mask),
            ("test", ~train_mask & frame[DATE_COLUMN].notna()),
        ):
            chunk = frame.loc[mask]
            counts[part] += len(chunk)
            if len(chunk):
                samples[part], priorities[part] = _keep_smallest(
                    samples[part], priorities[part], chunk, rng.random(len(chunk)), sample_size
                )

    total = sum(counts.values()) + null_counts[DATE_COLUMN]
    audit = {
        "rows": total,
        "period_rows": counts,
        "null_counts": null_counts,
        "cutoff": cutoff.date().isoformat(),
        "sample_rows": {p: len(df) for p, df in samples.items()},
    }
    if not counts["train"] or not counts["test"]:
        raise ValueError(f"Both periods need data; counts: {counts}")
    return audit, samples["train"], samples["test"]


def validate(parquet_path: Path, report_dir: Path, cutoff: pd.Timestamp, sample_size: int):
    """Run data-quality checks and save their reports and an audit summary.

    First scans the Parquet file and prepares bounded train and test samples.
    Deepchecks evaluates null percentages, mixed data types, duplicate rows,
    and feature drift. The suite requires no nulls, limits the duplicate ratio
    to 1%, and limits categorical and numeric drift scores to 0.2. The station
    ID columns are treated as categorical features. The report directory is
    created if needed, and the result is written to ``deepchecks.html``,
    ``deepchecks.json``, and ``audit.json``. The audit summary records whether
    all Deepchecks checks passed.

    Args:
        parquet_path: Parquet file to audit.
        report_dir: Directory where the HTML, JSON, and audit reports are saved.
        cutoff: Timestamp separating the train and test periods.
        sample_size: Maximum number of rows sampled from each period.

    Returns:
        The audit dictionary, including row and null counts, sample sizes, and
        the Deepchecks pass status.

    Raises:
        RuntimeError: If Deepchecks cannot be imported in the current environment.
        ValueError: If the Parquet schema is invalid or either period has no rows.
    """
    audit, train_frame, test_frame = scan_and_sample(parquet_path, cutoff, sample_size)
    try:
        from deepchecks.tabular import Dataset, Suite
        from deepchecks.tabular.checks import (
            DataDuplicates,
            FeatureDrift,
            MixedDataTypes,
            PercentOfNulls,
        )
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            "Deepchecks could not load. Use Python 3.10 and install "
            "requirements-deepchecks.txt in a separate virtual environment."
        ) from error

    train = Dataset(train_frame, cat_features=["origen_id", "destino_id"], dataset_name="Train")
    test = Dataset(test_frame, cat_features=["origen_id", "destino_id"], dataset_name="Test")
    suite = Suite(
        "MiBici trip data quality",
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0),
        MixedDataTypes(),
        DataDuplicates().add_condition_ratio_less_or_equal(0.01),
        FeatureDrift(
            columns=list(DRIFT_COLUMNS), n_samples=sample_size
        ).add_condition_drift_score_less_than(
            max_allowed_categorical_score=0.2, max_allowed_numeric_score=0.2
        ),
    )
    result = suite.run(train_dataset=train, test_dataset=test)
    report_dir.mkdir(parents=True, exist_ok=True)
    result.save_as_html(str(report_dir / "deepchecks.html"), as_widget=False)
    (report_dir / "deepchecks.json").write_text(result.to_json(), encoding="utf-8")
    audit["deepchecks_passed"] = result.passed(fail_if_check_not_run=True)
    (report_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    """Run validation from the command line using parsed arguments.

    Accepts ``--parquet`` and ``--report-dir`` paths, a ``--cutoff`` timestamp,
    and ``--sample-size``. Defaults come from ``DEFAULT_PARQUET`` and
    ``DEFAULT_REPORT_DIR``, July 1, 2026, and 5,000 rows per period; the CLI
    requires the sample size to be at least 10 and the Parquet path to exist.
    It prints the resulting audit summary and report path. The process exits
    with status 1 if any required column contains nulls or a Deepchecks check
    fails; ``argparse`` reports invalid arguments and exits with an error code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cutoff", type=pd.Timestamp, default=pd.Timestamp("2026-07-01"))
    parser.add_argument("--sample-size", type=int, default=5_000)
    args = parser.parse_args()
    if args.sample_size < 10:
        parser.error("--sample-size must be at least 10")
    if not args.parquet.is_file():
        parser.error(f"Parquet file not found: {args.parquet}")
    audit = validate(args.parquet, args.report_dir, args.cutoff, args.sample_size)
    print(json.dumps(audit, indent=2))
    print(f"Deepchecks report: {args.report_dir / 'deepchecks.html'}")
    if any(audit["null_counts"].values()) or not audit["deepchecks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
