from pathlib import Path
import typer
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from mi_bici_experiment.config import FIGURES_DIR, PROCESSED_DATA_DIR

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

if __name__ == "__main__":
    app()
