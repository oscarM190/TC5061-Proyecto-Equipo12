from pathlib import Path

import typer
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from mi_bici_experiment.config import FIGURES_DIR, PROCESSED_DATA_DIR

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402
from loguru import logger
import matplotlib

matplotlib.use("Agg")
app = typer.Typer()

def _guardar(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path

def print_balance_normalizado(balance_diario):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        balance_diario["Balance_Normalizado"],
        bins=100,
        kde=True,
        ax=ax,
    )
    ax.set_title("Distribución del Balance Diario Normalizado")
    ax.set_xlabel("Balance Normalizado (Llegadas - Salidas) / Capacidad")
    ax.set_ylabel("Frecuencia (Días-Estación)")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)

    return _guardar(fig, FIGURES_DIR / "balance_normalizado.png")
    
def matriz_confucion_regresion_multiclass_clasificacion(y_test, y_pred, conjunto=None):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Sin Bicis', 'Balanceado', 'Sin Anclajes'],
                yticklabels=['Sin Bicis', 'Balanceado', 'Sin Anclajes'])
    title = 'Matriz de Confusión - Regresión Logística Multinomial'
    if conjunto is not None:
        title += f' - {conjunto}'
    ax.set_title(title)
    ax.set_xlabel('Predicción del Modelo')
    ax.set_ylabel('Valor Real (Ground Truth)')

    suffix = ""
    if conjunto is not None:
        suffix = "_" + "".join(
            character if character.isalnum() else "_"
            for character in conjunto.lower()
        )
    return _guardar(fig, FIGURES_DIR / f"matriz_confusion_multinomial{suffix}.png")

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
