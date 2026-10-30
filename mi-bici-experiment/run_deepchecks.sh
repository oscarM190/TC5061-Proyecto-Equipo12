#!/usr/bin/env bash
set -euo pipefail

# Keep the Linux environment separate from the Windows one in shared checkouts.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$project_root/.." && pwd)"
environment_path="$project_root/.venv-deepchecks-linux"
python_path="$environment_path/bin/python"
local_uv_path="$repository_root/.venv/bin/uv"

if [[ -x "$local_uv_path" ]]; then
    uv_path="$local_uv_path"
elif command -v uv >/dev/null 2>&1; then
    uv_path="$(command -v uv)"
else
    printf 'uv was not found in the main .venv or on PATH. Install uv first.\n' >&2
    exit 1
fi

if [[ -x "$python_path" ]]; then
    printf 'Using existing Python 3.10 environment: %s\n' "$environment_path"
elif [[ -e "$environment_path" || -L "$environment_path" ]]; then
    printf 'An incomplete environment exists at %s. Remove or repair it before retrying.\n' "$environment_path" >&2
    exit 1
else
    printf 'Creating Python 3.10 environment: %s\n' "$environment_path"
    "$uv_path" venv --no-project --managed-python --python 3.10 "$environment_path"
fi

"$python_path" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'

printf 'Installing Deepchecks requirements...\n'
"$uv_path" pip install --python "$python_path" -r "$project_root/requirements-deepchecks.txt"

cd -- "$project_root"
printf 'Running data validation tests...\n'
"$python_path" -m pytest -q tests/test_data_validation.py

printf 'Validating the trips Parquet with Deepchecks...\n'
"$python_path" -m mi_bici_experiment.data_validation

printf 'Validating the P3 station profiles with Deepchecks...\n'
if [[ -f data/processed/p3_perfil_train.parquet ]]; then
    "$python_path" -m mi_bici_experiment.validacion_p3 validar
else
    printf 'P3 profiles not found. Run first: python -m mi_bici_experiment.validacion_p3 exportar\n' >&2
    exit 1
fi
