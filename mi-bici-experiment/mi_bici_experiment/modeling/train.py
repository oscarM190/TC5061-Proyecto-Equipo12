from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from mi_bici_experiment.config import MODELS_DIR, PROCESSED_DATA_DIR
from mi_bici_experiment.modeling.modules.p1_regression.models import RegressionModel
from mi_bici_experiment.modeling.modules.p2_classification.models import (
    ClassificationModel,
)
from mi_bici_experiment.modeling.modules.p3_clustering.models import ClusteringModel

app = typer.Typer()


# Ejecuta los tres módulos en secuencia.
@app.command()
def main():
    """Ejecuta P1, P2 y P3 en secuencia."""
    RegressionModel().ejecutar()
    ClassificationModel.ejecutar()
    ClusteringModel().ejecutar()


if __name__ == "__main__":
    main()
