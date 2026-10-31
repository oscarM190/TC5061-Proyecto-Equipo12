"""Chequeos de seguridad del repositorio: nada sensible versionado y .gitignore efectivo.

Complementan a bandit (código) y pip-audit (dependencias); ver README, "Análisis de seguridad".
"""

import re
import subprocess

import pytest

from mi_bici_experiment.config import PROJ_ROOT

REPO = PROJ_ROOT.parent


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


if git("rev-parse", "--is-inside-work-tree").returncode != 0:
    pytest.skip("Los chequeos de seguridad requieren un clon de Git", allow_module_level=True)

ARCHIVOS = git("ls-files").stdout.splitlines()

# Patrones armados por partes para que este archivo no se detecte a sí mismo.
SECRETOS = {
    "token de GitHub": r"\b(gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{50,})",
    "llave de AWS": r"\bAKIA[0-9A-Z]{16}\b",
    "llave privada": r"-----BEGIN [A-Z ]*PRIVATE" + r" KEY-----",
    "credencial en una URL": r"https?://[^/\s:@'\"]+:[^/\s@'\"]+@",
    "contraseña escrita": r"(?i)\b(password|passwd|contraseña|secret|api_key|token)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]",
}
EXTENSIONES_TEXTO = (".py", ".md", ".yml", ".yaml", ".toml", ".cfg", ".txt", ".json", ".sh",
                     ".ps1", ".ipynb", ".example", "MLproject", "Makefile")


@pytest.mark.parametrize(
    "patron",
    [
        r"(^|/)\.env$",
        r"\.(db|sqlite3?)$",
        r"(^|/)(mlruns|mlartifacts)/",
        r"\.(pkl|joblib|skops|onnx)$",
        r"(^|/)data/.+\.(csv|parquet)$",
    ],
    ids=["env", "bases_de_datos", "carpetas_mlflow", "modelos", "datos_mibici"],
)
def test_sin_archivos_sensibles_versionados(patron):
    """Ni credenciales (.env), ni MLflow local, ni modelos, ni datos de MiBici en Git."""
    assert [a for a in ARCHIVOS if re.search(patron, a)] == []


@pytest.mark.parametrize(
    "ruta",
    [
        ".env",
        "mi-bici-experiment/.env",
        "mi-bici-experiment/data/external/datos_abiertos_2026_09.csv",
        "mi-bici-experiment/data/processed/p3_perfil_train.parquet",
        "mi-bici-experiment/mlflow.db",
        "mlflow.db",
        "mi-bici-experiment/mlruns/1/meta.yaml",
        "mi-bici-experiment/mlartifacts/1/model.pkl",
        "mi-bici-experiment/models/p3.joblib",
        "mi-bici-experiment/reports/deepchecks/audit.json",
    ],
)
def test_gitignore_excluye(ruta):
    """Si alguien genera estos archivos, `git add .` no los sube."""
    assert git("check-ignore", "-q", "--no-index", ruta).returncode == 0, ruta


def test_env_example_sin_valores_secretos():
    """La plantilla documenta las variables de entorno sin traer credenciales."""
    plantilla = PROJ_ROOT / ".env.example"
    assert plantilla.is_file()
    valores = dict(
        linea.split("=", 1) for linea in plantilla.read_text().splitlines()
        if "=" in linea and not linea.startswith("#")
    )
    assert valores["MLFLOW_TRACKING_PASSWORD"] == ""
    assert valores["MLFLOW_TRACKING_USERNAME"] == ""


def test_sin_secretos_en_archivos_versionados():
    hallazgos = []
    for archivo in ARCHIVOS:
        ruta = REPO / archivo
        if not archivo.endswith(EXTENSIONES_TEXTO) or not ruta.is_file():
            continue
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        hallazgos += [f"{archivo}: {nombre}" for nombre, patron in SECRETOS.items()
                      if re.search(patron, texto)]
    assert hallazgos == []
