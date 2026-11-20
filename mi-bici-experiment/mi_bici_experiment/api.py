from contextlib import asynccontextmanager
from functools import lru_cache
import os
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from mi_bici_experiment.config import P1_NOMBRE_REGISTRO

DEFAULT_TRACKING_URI = "http://localhost:5000"
MODEL_ALIAS = "champion"
# Cola de RQ para predicciones asíncronas (opcional: solo si existe REDIS_URL).
QUEUE_NAME = "predicciones"
JOB_TTL_SECONDS = 3600
MODEL_FEATURES = (
    "trips_lag_1",
    "trips_lag_7",
    "trips_roll_mean_7",
    "is_weekend",
    "origen_id",
    "dayofweek",
    "month",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Con MIBICI_PRELOAD_MODEL=1 descarga el modelo al arrancar y no en la primera petición."""

    if os.getenv("MIBICI_PRELOAD_MODEL") == "1":
        get_model()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="MiBici Demand API",
    version="1.0.0",
    description="Predice los viajes diarios de salida por estación con el modelo P1 de MiBici.",
)


class PredictionRequest(BaseModel):
    """Variables usadas por el modelo P1 para una estación y un día."""

    model_config = ConfigDict(extra="forbid")

    trips_lag_1: float = Field(ge=0, description="Viajes de la estación el día anterior.")
    trips_lag_7: float = Field(ge=0, description="Viajes de la estación hace siete días.")
    trips_roll_mean_7: float = Field(
        ge=0, description="Media de viajes de los siete días previos."
    )
    is_weekend: Literal[0, 1] = Field(
        description="1 si el día es sábado o domingo; 0 en otro caso."
    )
    origen_id: int = Field(gt=0, description="Identificador de la estación de origen.")
    dayofweek: int = Field(ge=0, le=6, description="Día de la semana (lunes=0, domingo=6).")
    month: int = Field(ge=1, le=12, description="Mes del año.")


class AsyncPredictionResponse(BaseModel):
    """Estado de una predicción encolada en Redis Queue."""

    job_id: str = Field(description="Identificador del trabajo en la cola.")
    status: str = Field(description="queued, started, finished o failed.")
    prediction: float | None = Field(default=None, description="Viajes estimados al terminar.")
    model: str = Field(description="Modelo registrado que produce la predicción.")


class PredictionResponse(BaseModel):
    """Prediction returned by the P1 model."""

    prediction: float = Field(description="Número estimado de viajes de salida diarios.")
    model: str = Field(description="Modelo registrado que produjo la predicción.")


def model_uri() -> str:
    """Return the URI for the model version that is approved for serving."""

    alias = os.getenv("MIBICI_MODEL_ALIAS", MODEL_ALIAS)
    return f"models:/{P1_NOMBRE_REGISTRO}@{alias}"


@lru_cache
def get_model():
    """Load the registered model once per API process, on its first prediction request."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    uri = model_uri()
    logger.info("Loading MiBici model from {}", uri)
    return mlflow.sklearn.load_model(uri)


@lru_cache
def get_queue():
    """Cola de RQ si la API corre con Redis (docker compose); None si no hay REDIS_URL."""

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    from redis import Redis
    from rq import Queue

    return Queue(QUEUE_NAME, connection=Redis.from_url(redis_url))


def predict_values(values: dict) -> float:
    """Predicción del modelo para un diccionario con las siete variables; la usan la API y el worker."""

    features = pd.DataFrame([values], columns=MODEL_FEATURES)
    # A demand count cannot be negative; this mirrors P1's evaluation behavior.
    return max(0.0, float(get_model().predict(features)[0]))


def require_queue():
    queue = get_queue()
    if queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Las predicciones asíncronas requieren Redis (REDIS_URL).",
        )
    return queue


@app.get("/health")
def health() -> dict[str, str]:
    """Expose a lightweight check that does not download the model."""

    return {"status": "ok", "model_uri": model_uri()}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict daily departures for one station from the model's seven input variables."""

    try:
        prediction = predict_values(request.model_dump())
    except Exception as error:
        logger.exception("Unable to generate a MiBici prediction")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de predicción no está disponible.",
        ) from error

    return PredictionResponse(prediction=prediction, model=model_uri())


@app.post(
    "/predict/async",
    response_model=AsyncPredictionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def predict_async(request: PredictionRequest) -> AsyncPredictionResponse:
    """Encola la predicción para el worker de RQ y responde de inmediato con el job_id."""

    job = require_queue().enqueue(
        "mi_bici_experiment.api.predict_values",
        request.model_dump(),
        result_ttl=JOB_TTL_SECONDS,
        failure_ttl=JOB_TTL_SECONDS,
    )
    return AsyncPredictionResponse(job_id=job.id, status="queued", model=model_uri())


@app.get("/predict/async/{job_id}", response_model=AsyncPredictionResponse)
def predict_async_result(job_id: str) -> AsyncPredictionResponse:
    """Consulta el estado de una predicción encolada y su resultado cuando termina."""

    job = require_queue().fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trabajo no encontrado.")
    job_status = job.get_status()
    job_status = getattr(job_status, "value", job_status)
    prediction = job.return_value() if job_status == "finished" else None
    return AsyncPredictionResponse(
        job_id=job.id, status=job_status, prediction=prediction, model=model_uri()
    )
