FROM python:3.14.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/p3 \
    MLFLOW_ENABLE_TELEMETRY=false \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app
COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 appuser

COPY --chown=10001:10001 serving/ ./serving/
COPY --chown=10001:10001 artifacts/p3/ ./artifacts/p3/

USER 10001:10001
# Verifica los archivos y la firma durante la construcción; no conecta al Registry.
RUN python -c "from pathlib import Path; from serving.model import load_bundle; load_bundle(Path('/app/artifacts/p3'))"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
