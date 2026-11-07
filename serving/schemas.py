"""Contrato HTTP: perfiles agregados por estación, nunca viajes individuales."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

FEATURES = ("prop_fin_semana", "log_viajes_dia", "balance")


class StationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    prop_fin_semana: float = Field(ge=0, le=1, description="Salidas de fin de semana / salidas totales.")
    log_viajes_dia: float = Field(ge=0, description="log1p del promedio diario de salidas.")
    balance: float = Field(ge=-1, le=1, description="(Llegadas − salidas) / (llegadas + salidas).")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[StationProfile] = Field(min_length=1, max_length=1000)


class ModelIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^[1-9][0-9]*$")
    run_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: ModelIdentity
    predictions: list[Annotated[int, Field(ge=0, strict=True)]] = Field(min_length=1, max_length=1000)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: ModelIdentity
