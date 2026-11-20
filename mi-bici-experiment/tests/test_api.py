import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from mi_bici_experiment import api


# Stand in for the registered model and capture the features passed to predict.
class DummyModel:
    def __init__(self, prediction: float = 12.5):
        self.prediction = prediction
        self.features = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.features = features
        return np.array([self.prediction])


# Build a valid prediction request with every expected model feature.
def prediction_payload() -> dict:
    return {
        "trips_lag_1": 12,
        "trips_lag_7": 9,
        "trips_roll_mean_7": 10.5,
        "is_weekend": 0,
        "origen_id": 42,
        "dayofweek": 2,
        "month": 7,
    }


# [TEST 1] A valid request uses the registered model and passes features in model order.
def test_predict_uses_registered_model_with_expected_feature_order(monkeypatch):
    model = DummyModel()
    monkeypatch.setattr(api, "get_model", lambda: model)

    response = TestClient(api.app).post("/predict", json=prediction_payload())

    assert response.status_code == 200
    assert response.json()["prediction"] == 12.5
    assert response.json()["model"] == api.model_uri()
    assert list(model.features.columns) == list(api.MODEL_FEATURES)
    assert model.features.iloc[0].to_dict() == prediction_payload()


# [TEST 2] A request with an invalid model feature is rejected with status 422.
def test_predict_rejects_invalid_model_features():
    payload = prediction_payload() | {"dayofweek": 7}

    response = TestClient(api.app).post("/predict", json=payload)

    assert response.status_code == 422


# [TEST 3] A negative regression result is clipped to zero in the response.
def test_predict_clips_a_negative_regression_result(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: DummyModel(-2.0))

    response = TestClient(api.app).post("/predict", json=prediction_payload())

    assert response.status_code == 200
    assert response.json()["prediction"] == 0.0


# Stand in for an RQ job and queue so the async endpoints run without Redis.
class FakeJob:
    def __init__(self, job_id: str, status: str = "queued", result=None):
        self.id = job_id
        self.status = status
        self.result = result

    def get_status(self) -> str:
        return self.status

    def return_value(self):
        return self.result


class FakeQueue:
    def __init__(self):
        self.jobs = {}
        self.enqueued = []

    def enqueue(self, func, values, **kwargs):
        self.enqueued.append((func, values))
        job = FakeJob(f"job-{len(self.enqueued)}")
        self.jobs[job.id] = job
        return job

    def fetch_job(self, job_id):
        return self.jobs.get(job_id)


# [TEST 4] The async endpoint enqueues the validated features for the RQ worker.
def test_predict_async_enqueues_the_prediction(monkeypatch):
    queue = FakeQueue()
    monkeypatch.setattr(api, "get_queue", lambda: queue)

    response = TestClient(api.app).post("/predict/async", json=prediction_payload())

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert queue.enqueued == [("mi_bici_experiment.api.predict_values", prediction_payload())]


# [TEST 5] A finished job returns the worker's prediction; an unknown job returns 404.
def test_predict_async_result_returns_finished_prediction(monkeypatch):
    queue = FakeQueue()
    queue.jobs["listo"] = FakeJob("listo", status="finished", result=7.5)
    monkeypatch.setattr(api, "get_queue", lambda: queue)
    client = TestClient(api.app)

    finished = client.get("/predict/async/listo")
    missing = client.get("/predict/async/no-existe")

    assert finished.status_code == 200
    assert finished.json()["prediction"] == 7.5
    assert missing.status_code == 404


# [TEST 6] Without Redis the async endpoint reports 503 instead of failing.
def test_predict_async_without_redis_returns_503(monkeypatch):
    monkeypatch.setattr(api, "get_queue", lambda: None)

    response = TestClient(api.app).post("/predict/async", json=prediction_payload())

    assert response.status_code == 503


# [TEST 7] The worker's function applies the same model and clipping as /predict.
def test_predict_values_matches_the_sync_endpoint(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: DummyModel(-1.0))

    assert api.predict_values(prediction_payload()) == 0.0
