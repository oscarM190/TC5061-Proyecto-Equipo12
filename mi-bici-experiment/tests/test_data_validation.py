import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mi_bici_experiment.data_validation import check_schema, scan_and_sample


# [TEST 1] Scanning counts every row, records nulls, and samples train and test at the cutoff.
def test_scan_counts_all_rows_and_splits_at_cutoff(tmp_path):
    schema = pa.schema(
        [
            ("inicio_del_viaje", pa.timestamp("us")),
            ("fin_del_viaje", pa.timestamp("us")),
            ("origen_id", pa.int16()),
            ("destino_id", pa.int16()),
            ("longitude", pa.float32()),
            ("latitude", pa.float32()),
            ("dpcapacity", pa.int8()),
        ]
    )
    table = pa.Table.from_pydict(
        {
            "inicio_del_viaje": [
                pd.Timestamp("2026-06-29"), pd.Timestamp("2026-06-30"),
                pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02"),
            ],
            "fin_del_viaje": [
                pd.Timestamp("2026-06-29"), pd.Timestamp("2026-06-30"),
                pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-02"),
            ],
            "origen_id": [1, 2, 3, 4],
            "destino_id": [2, 3, 4, 5],
            "longitude": [-103.3] * 4,
            "latitude": [20.7] * 4,
            "dpcapacity": [20, None, 30, 40],
        },
        schema=schema,
    )
    path = tmp_path / "trips.parquet"
    pq.write_table(table, path)

    audit, train, test = scan_and_sample(path, pd.Timestamp("2026-07-01"), 10, batch_size=2)

    assert audit["rows"] == 4
    assert audit["period_rows"] == {"train": 2, "test": 2}
    assert audit["null_counts"]["dpcapacity"] == 1
    assert train["inicio_del_viaje"].max() < pd.Timestamp("2026-07-01")
    assert test["inicio_del_viaje"].min() >= pd.Timestamp("2026-07-01")
    limited, train_one, test_one = scan_and_sample(
        path, pd.Timestamp("2026-07-01"), 1, batch_size=2
    )
    assert limited["sample_rows"] == {"train": 1, "test": 1}
    assert len(train_one) == len(test_one) == 1


# [TEST 2] The schema check rejects unexpected column types and missing required columns.
def test_schema_rejects_wrong_types():
    schema = pa.schema([
        ("inicio_del_viaje", pa.string()),
        ("fin_del_viaje", pa.timestamp("us")),
        ("origen_id", pa.int16()),
        ("destino_id", pa.int16()),
        ("longitude", pa.float32()),
        ("latitude", pa.float32()),
        ("dpcapacity", pa.int8()),
    ])
    with pytest.raises(ValueError, match="Unexpected Parquet types"):
        check_schema(schema)
    with pytest.raises(ValueError, match="Missing required columns"):
        check_schema(pa.schema([("inicio_del_viaje", pa.timestamp("us"))]))
