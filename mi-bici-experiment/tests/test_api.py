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
