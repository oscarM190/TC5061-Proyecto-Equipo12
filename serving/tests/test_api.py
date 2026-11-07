import json
from pathlib import Path
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

import serving.app as api
from serving.model import LoadedModel
from serving.schemas import FEATURES, ModelIdentity

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


PROFILE = {"prop_fin_semana": 0.2, "log_viajes_dia": 3.0, "balance": -0.1}
IDENTITY = ModelIdentity(name="mibici-equipo12-p3-grupos", version="1", run_id="test-run",
                         uri="models:/mibici-equipo12-p3-grupos/1")


@pytest.fixture
async def client(monkeypatch):
    calls = []

    def predict(frame):
        calls.append(frame)
        return np.arange(len(frame), dtype="int32") % 3

    loaded = LoadedModel(SimpleNamespace(predict=predict), IDENTITY)
    loads = []

    def load(path):
        loads.append(path)
        return loaded

    monkeypatch.setattr(api, "load_bundle", load)
    application = api.create_app(Path("unused"))
    async with application.router.lifespan_context(application):
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as http:
            yield http, calls, loads, loaded


async def test_prediction_contract_order_and_single_load(client):
    http, calls, loads, _ = client
    body = {"instances": [PROFILE, {**PROFILE, "balance": 1}]}
    for _ in range(2):
        result = await http.post("/predict", json=body)
        assert result.status_code == 200
        assert result.json() == {"model": IDENTITY.model_dump(), "predictions": [0, 1]}
    assert len(loads) == 1
    assert list(calls[0].columns) == list(FEATURES)
    assert calls[0].dtypes.astype(str).eq("float64").all()
    assert calls[0].balance.tolist() == [-0.1, 1]
    assert (await http.get("/healthz")).json()["status"] == "ok"
    assert (await http.get("/model")).json() == IDENTITY.model_dump()
    assert (await http.get("/docs")).status_code == 200


@pytest.mark.parametrize("body", [
    {}, {"instances": []}, {"instances": [PROFILE] * 1001},
    {"instances": [PROFILE], "extra": 1},
    {"instances": [{**PROFILE, "extra": 1}]},
    {"instances": [{"prop_fin_semana": 0.2, "balance": 0.0}]},
    *[{"instances": [{**PROFILE, key: value}]} for key, value in [
        ("prop_fin_semana", -0.1), ("prop_fin_semana", 1.1),
        ("log_viajes_dia", -1.0), ("balance", -1.1), ("balance", 1.1),
        ("balance", "0.1"), ("balance", True), ("balance", None),
        ("balance", []), ("prop_fin_semana", "NaN"), ("log_viajes_dia", "Infinity"),
    ]],
])
async def test_invalid_requests_do_not_reach_model(client, body):
    http, calls, _, _ = client
    assert (await http.post("/predict", json=body)).status_code == 422
    assert calls == []


async def test_boundary_values_are_valid(client):
    http, _, _, _ = client
    result = await http.post("/predict", json={"instances": [
        {"prop_fin_semana": 0, "log_viajes_dia": 0, "balance": -1},
        {"prop_fin_semana": 1, "log_viajes_dia": 0, "balance": 1},
    ]})
    assert result.status_code == 200


async def test_invalid_json(client):
    http, calls, _, _ = client
    result = await http.post("/predict", content="{", headers={"Content-Type": "application/json"})
    assert result.status_code == 422
    assert not calls


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
async def test_nonstandard_json_numbers_return_422(client, value):
    http, calls, _, _ = client
    body = '{"instances":[{"prop_fin_semana":0.2,"log_viajes_dia":' + value + ',"balance":0.0}]}'
    result = await http.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert result.status_code == 422
    assert not calls


@pytest.mark.parametrize("output", [np.array([]), np.array([1.5]), np.array([-1]),
                                   np.array([True]), np.array([[1]]), np.array(["1"])])
async def test_invalid_model_output_is_not_reported_as_success(client, output):
    http, _, _, loaded = client
    loaded.predictor.predict = lambda frame: output
    result = await http.post("/predict", json={"instances": [PROFILE]})
    assert result.status_code == 500
    assert result.json() == {"detail": "No se pudo completar la predicción."}


async def test_internal_error_does_not_expose_paths_or_credentials(client):
    http, _, _, loaded = client

    def fail(frame):
        raise RuntimeError("ruta-interna-privada")

    loaded.predictor.predict = fail
    result = await http.post("/predict", json={"instances": [PROFILE]})
    assert result.status_code == 500
    assert "ruta-interna" not in result.text


async def test_missing_model_prevents_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        application = api.create_app(tmp_path)
        async with application.router.lifespan_context(application):
            pass


async def test_real_registry_bundle():
    """Integración opcional en CI; localmente usa la versión descargada del Registry real."""
    root = Path(__file__).resolve().parents[2]
    bundle = root / "artifacts/p3"
    if not bundle.exists():
        pytest.skip("Exporta la versión real del Registry para la prueba de integración.")
    from serving.model import load_bundle
    from serving.schemas import StationProfile

    body = json.loads((root / "examples/p3-request.json").read_text())
    direct = load_bundle(bundle).predict([StationProfile(**p) for p in body["instances"]])
    application = api.create_app(bundle)
    async with application.router.lifespan_context(application):
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as http:
            result = await http.post("/predict", json=body)
            assert result.status_code == 200
            assert result.json()["predictions"] == direct
            assert result.json()["model"]["name"] == "mibici-equipo12-p3-grupos"
