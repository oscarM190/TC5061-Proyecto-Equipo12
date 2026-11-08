# Containerfile compatible con el estándar OCI (Podman / Docker)
# Empaqueta la API REST de inferencia de MiBici (Modelo P1 de Regresión)
FROM python:3.14-slim

# Dependencias del sistema necesarias para LightGBM y compilación C/C++
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_ROOT_USER_ACTION=ignore \
    MLFLOW_TRACKING_URI=http://host.containers.internal:5000 \
    MIBICI_MODEL_ALIAS=champion

WORKDIR /app

# Preinstalación de dependencias en capa de construcción para reproducibilidad
COPY mi-bici-experiment/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copia del código fuente de la aplicación
COPY mi-bici-experiment/mi_bici_experiment ./mi_bici_experiment

# Puerto de servicio
EXPOSE 8000

# Verificación de salud del contenedor
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio del servidor ASGI Uvicorn
CMD ["uvicorn", "mi_bici_experiment.api:app", "--host", "0.0.0.0", "--port", "8000"]
