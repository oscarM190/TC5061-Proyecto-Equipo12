"""Ejecuta la clasificación multinomial (P3) y la regresión (P1)."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    from models.clasificacion_regresion_lineal_multinomial import RegresionLinealMultinomial
    from mi_bici_experiment.modeling.train import RegresionP1

    print("Iniciando clasificación multinomial (P3)...", flush=True)
    RegresionLinealMultinomial.run()

    print("Iniciando regresión (P1)...", flush=True)
    RegresionP1().ejecutar()

    print("Entrenamiento de ambos modelos completado.", flush=True)


if __name__ == "__main__":
    main()
