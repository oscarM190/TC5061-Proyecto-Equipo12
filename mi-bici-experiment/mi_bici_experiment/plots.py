from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402
import typer  # noqa: E402

from mi_bici_experiment.config import FIGURES_DIR, PROCESSED_DATA_DIR  # noqa: E402

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


# ---------------------------------------------------------------------------------------------
# P3: gráficas de los grupos de estaciones (se registran como artefactos en MLflow)
# ---------------------------------------------------------------------------------------------

COLORES = plt.get_cmap("tab10").colors


def _guardar(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def mapa_grupos(estaciones: pd.DataFrame, titulo: str, path: Path) -> Path:
    """Estaciones en su ubicación, coloreadas por grupo: muestra dónde se concentra cada patrón."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for g, d in estaciones.groupby("grupo"):
        ax.scatter(d["lon"], d["lat"], s=14, color=COLORES[g % 10], label=f"Grupo {g} ({len(d)})")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    return _guardar(fig, path)


def perfil_grupos(perfil: pd.DataFrame, path: Path) -> Path:
    """Perfil horario medio de salidas y llegadas en días hábiles de cada grupo."""
    sal = [c for c in perfil.columns if c.startswith("sal_h")]
    lle = [c for c in perfil.columns if c.startswith("lle_h")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for g, d in perfil.groupby("grupo"):
        axes[0].plot(range(24), d[sal].mean().values, color=COLORES[g % 10], label=f"Grupo {g}")
        axes[1].plot(range(24), d[lle].mean().values, color=COLORES[g % 10])
    axes[0].set_title("Salidas en días hábiles")
    axes[1].set_title("Llegadas en días hábiles")
    for ax in axes:
        ax.set_xlabel("Hora del día")
    axes[0].set_ylabel("Proporción del día")
    axes[0].legend(fontsize=8)
    fig.suptitle("P3: patrón horario medio por grupo")
    return _guardar(fig, path)


def silueta_por_k(ks: list[int], valores: list[float], path: Path) -> Path:
    """Silueta de entrenamiento para cada número de grupos candidato; el máximo define k."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ks, valores, marker="o")
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Silueta (train)")
    ax.set_title("P3: selección de k")
    ax.grid(alpha=0.3)
    return _guardar(fig, path)


if __name__ == "__main__":
    app()
