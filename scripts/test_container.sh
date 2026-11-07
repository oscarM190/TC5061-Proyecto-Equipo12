#!/usr/bin/env bash
# Construye y prueba un contenedor real. Solo escribe evidencia después de cada ejecución.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
engine="${CONTAINER_ENGINE:-docker}"
image_name="${IMAGE_NAME:-mibici-p3:1}"
port="${PORT:-8000}"
container_name="mibici-p3-check-$$"
evidence_dir="${EVIDENCE_DIR:-evidence/container}"
mkdir -p "$evidence_dir"
command -v "$engine" >/dev/null
command -v curl >/dev/null
"$engine" info >"$evidence_dir/engine.txt" 2>&1 || {
    cat "$evidence_dir/engine.txt" >&2
    printf 'El motor de contenedores no está disponible; no se generó evidencia de ejecución.\n' >&2
    exit 1
}
test -f artifacts/p3/deployment.json || {
    printf 'Descarga primero el modelo con python -m serving.export_model.\n' >&2
    exit 1
}
"$engine" build -f Containerfile -t "$image_name" . 2>&1 | tee "$evidence_dir/build.txt"
cleanup() {
    "$engine" logs "$container_name" >"$evidence_dir/logs.txt" 2>&1 || true
    "$engine" rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
"$engine" run -d --name "$container_name" --read-only --tmpfs /tmp:rw,nosuid,size=128m \
    --cap-drop ALL --security-opt no-new-privileges \
    -p "127.0.0.1:$port:8000" "$image_name" >"$evidence_dir/container-id.txt"
base_url="http://127.0.0.1:$port"
ready=false
for attempt in {1..30}; do
    if curl --silent --fail "$base_url/healthz" >"$evidence_dir/health.json"; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    printf 'El servicio no estuvo listo en 30 segundos. Revisa logs.txt.\n' >&2
    exit 1
fi
cp examples/p3-request.json "$evidence_dir/request.json"
curl --silent --show-error --fail -H 'Content-Type: application/json' \
    --data-binary @examples/p3-request.json "$base_url/predict" >"$evidence_dir/response.json"
status="$(curl --silent --show-error -o "$evidence_dir/invalid-response.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' -d '{"instances":[]}' "$base_url/predict")"
test "$status" = 422
"$engine" inspect "$container_name" >"$evidence_dir/container-inspect.json"
"$engine" exec "$container_name" python -c \
    'from importlib.metadata import version; import json; print(json.dumps({p:version(p) for p in ["mlflow","fastapi","pydantic","numpy","pandas","scikit-learn"]},indent=2))' \
    >"$evidence_dir/versions.json"
# Comprueba contrato, versión y coherencia con las etiquetas calculadas por el artefacto.
"$engine" exec -i "$container_name" python -c '
import json,sys
from pathlib import Path
from serving.model import load_bundle
from serving.schemas import StationProfile,PredictionResponse
request,response=json.load(sys.stdin)
parsed=PredictionResponse.model_validate(response)
loaded=load_bundle(Path("/app/artifacts/p3"))
assert parsed.model == loaded.identity
assert parsed.predictions == loaded.predict([StationProfile(**x) for x in request["instances"]])
print("Predicciones HTTP iguales a las del modelo registrado; entrada inválida: HTTP 422.")
' < <(printf '[%s,%s]' "$(cat "$evidence_dir/request.json")" "$(cat "$evidence_dir/response.json")") \
    | tee "$evidence_dir/verification.txt"
if [[ "${CAPTURE_SCREENSHOT:-0}" == 1 ]]; then
    node tools/evidence/capture.mjs "$base_url" "$evidence_dir/api-container.png"
fi
printf 'Prueba real terminada. Evidencia: %s\n' "$evidence_dir"
printf 'Para generar también la captura, instala tools/evidence y usa CAPTURE_SCREENSHOT=1.\n'
