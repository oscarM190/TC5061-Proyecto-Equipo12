"""FastAPI: carga una sola vez el pipeline completo de P3 durante el arranque."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from serving.model import load_bundle
from serving.schemas import HealthResponse, ModelIdentity, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


def create_app(model_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        path = model_path or Path(os.environ.get("MODEL_PATH", "artifacts/p3"))
        # Fallar al iniciar evita devolver predicciones simuladas si falta el modelo.
        application.state.model = load_bundle(path)
        logger.info("Modelo cargado: %s", application.state.model.identity.uri)
        yield
        del application.state.model

    root_path = os.getenv("FASTAPI_ROOT_PATH", "/p3")

    application = FastAPI(
        title="MiBici · Agrupación de estaciones (P3)",
        description="Asigna perfiles de estaciones a los grupos del modelo registrado en MLflow.",
        version="1.0.0",
        lifespan=lifespan,
        root_path=root_path,
        servers=[{"url": root_path, "description": "Production"}],
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Evitar que NaN/Infinity del cuerpo se vuelvan a serializar como JSON inválido.
        errors = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.get("/healthz", response_model=HealthResponse, tags=["Servicio"])
    def health(request: Request):
        return HealthResponse(model=request.app.state.model.identity)

    @application.get("/model", response_model=ModelIdentity, tags=["Servicio"])
    def model_info(request: Request):
        return request.app.state.model.identity

    @application.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
    def predict(body: PredictionRequest, request: Request):
        loaded = request.app.state.model
        try:
            predictions = loaded.predict(body.instances)
            return PredictionResponse(model=loaded.identity, predictions=predictions)
        except Exception:
            logger.exception("Falló la inferencia de P3")
            raise HTTPException(status_code=500, detail="No se pudo completar la predicción.") from None

    return application


app = create_app()
