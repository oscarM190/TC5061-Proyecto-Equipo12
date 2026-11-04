FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_ROOT_USER_ACTION=ignore \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

WORKDIR /app

COPY requirements.txt .
COPY mi_bici_experiment ./mi_bici_experiment

EXPOSE 8000

CMD ["sh", "-c", "pip install -r requirements.txt && (python -c 'import uvicorn' 2>/dev/null || pip install uvicorn) && exec uvicorn mi_bici_experiment.api:app --host 0.0.0.0 --port 8000"]