from pathlib import Path
import pandas as pd

import typer

from mi_bici_experiment.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
):
    trips_df = pd.read_parquet(input_path / "trips.parquet")
    nomenclatura_df = pd.read_parquet(input_path / "nomenclatura.parquet")

    features_df = trips_df[
        ["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
    ].merge(
        nomenclatura_df[["id", "latitude", "longitude", "dpcapacity"]].rename(
            columns={"id": "Origen_Id"}
        ),
        on="Origen_Id",
        how="left",
        validate="many_to_one",
    )

    print(features_df.info())

    output_path.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path / "features.parquet", index=False)


if __name__ == "__main__":
    app()
