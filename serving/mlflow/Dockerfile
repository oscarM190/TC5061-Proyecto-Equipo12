# Servidor de tracking y Model Registry de MLflow para docker-compose.yml.
# Backend SQLite y artefactos servidos por el propio servidor (--serve-artifacts), ambos en el
# volumen /mlflow para que el registro sobreviva a `docker compose down`. Solo acepta peticiones
# dirigidas a los nombres que usa Compose (mlflow) o al host local (protección contra DNS rebinding).
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 PIP_ROOT_USER_ACTION=ignore
RUN python -m pip install --no-cache-dir mlflow==3.17.2

RUN useradd --create-home --uid 1000 mlflow && mkdir -p /mlflow && chown mlflow /mlflow
USER mlflow
WORKDIR /mlflow
EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/mlartifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*"]
