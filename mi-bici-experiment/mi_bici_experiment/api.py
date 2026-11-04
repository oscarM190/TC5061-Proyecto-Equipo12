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
MODEL_FEATURES = (
    "trips_lag_1",
    "trips_lag_7",
    "trips_roll_mean_7",
    "is_weekend",
    "origen_id",
    "dayofweek",
    "month",
)

app = FastAPI(
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


@app.get("/health")
def health() -> dict[str, str]:
    """Expose a lightweight check that does not download the model."""

    return {"status": "ok", "model_uri": model_uri()}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict daily departures for one station from the model's seven input variables."""

    features = pd.DataFrame([request.model_dump()], columns=MODEL_FEATURES)
    try:
        prediction = float(get_model().predict(features)[0])
    except Exception as error:
        logger.exception("Unable to generate a MiBici prediction")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de predicción no está disponible.",
        ) from error

    # A demand count cannot be negative; this mirrors P1's evaluation behavior.
    return PredictionResponse(prediction=max(0.0, prediction), model=model_uri())
