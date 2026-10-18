"""Utilidades que comparten P1, P2 y P3 para ligar cada run de MLflow con su código y sus datos."""

import hashlib
import subprocess

from mi_bici_experiment.config import MI_BICI_CSVS, PROJ_ROOT, RAW_DATA_DIR


def git_info() -> dict:
    """Commit, rama y cambios sin commit; ligan cada run con el código que lo produjo."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(git("status", "--porcelain", "--untracked-files=no"))).lower(),
    }


def version_datos() -> str:
    """Huella SHA-256 de los CSV crudos usados (nombre y contenido), para registrarla en MLflow."""
    total = hashlib.sha256()
    for url in MI_BICI_CSVS:
        ruta = RAW_DATA_DIR / url.rsplit("/", 1)[-1]
        h = hashlib.sha256()
        with open(ruta, "rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
        total.update(f"{ruta.name}:{h.hexdigest()}".encode())
    return f"sha256-{total.hexdigest()[:12]}"
