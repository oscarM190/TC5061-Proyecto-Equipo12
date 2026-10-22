import logging
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import typer
from loguru import logger

app = typer.Typer()


def _execute_model(name: str, execute: Callable[[], object]) -> None:
    """Run a model quietly and print one message after successful completion."""
    previous_logging_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    try:
        # Capture prints, warnings, progress bars, and logging sent to the console.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            execute()
    finally:
        logging.disable(previous_logging_level)

    print(f"[INFO] Executed {name} model successfully!")


# Ejecuta los tres módulos en secuencia.
@app.command()
def main():
    """Ejecuta P1, P2 y P3 en secuencia."""
    # Disable project logs before importing modules that log during initialization.
    logger.disable("mi_bici_experiment")
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from mi_bici_experiment.modeling.modules.p1_regression.models import (
                RegressionModel,
            )
            from mi_bici_experiment.modeling.modules.p2_classification.models import (
                ClassificationModel,
            )
            from mi_bici_experiment.modeling.modules.p3_clustering.models import (
                ClusteringModel,
            )

        _execute_model("P1", lambda: RegressionModel().ejecutar())
        _execute_model("P2", ClassificationModel.ejecutar)
        _execute_model("P3", lambda: ClusteringModel().ejecutar())
    finally:
        logger.enable("mi_bici_experiment")


if __name__ == "__main__":
    main()
