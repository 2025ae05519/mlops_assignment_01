import json
from pathlib import Path

from fastapi.testclient import TestClient

from src import predict as predict_module
from src.api import app

client = TestClient(app)
SAMPLE_REQUEST = Path(__file__).resolve().parents[1] / "sample_request.json"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_returns_503_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "missing.joblib")
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_predict_rejects_missing_features():
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422


def test_predict_returns_prediction_and_confidence():
    payload = json.loads(SAMPLE_REQUEST.read_text())
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["positive_class_probability"] <= 1.0


def test_metrics_endpoint_exposes_request_counters():
    client.get("/health")
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert "heart_api_requests_total" in response.text
    assert "heart_api_predictions_total" in response.text
