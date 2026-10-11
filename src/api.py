"""FastAPI app used to serve the trained heart disease model.

Endpoints:
- GET  /health   readiness check (HTTP 503 when the model cannot be loaded)
- POST /predict  prediction with confidence
- GET  /metrics  Prometheus metrics for monitoring

Every request is logged (method, path, status, duration) and counted in
Prometheus metrics, so the container output and the /metrics endpoint can be
used as monitoring evidence during local, Docker or Kubernetes testing.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from src.predict import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Heart Disease Classification API", version="1.1.0")

REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "heart_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "heart_api_predictions_total",
    "Predictions returned by the model, by predicted label",
    ["label"],
)

# Prometheus scrapes this endpoint.
app.mount("/metrics", make_asgi_app())


class PredictionRequest(BaseModel):
    """Expected JSON body for the prediction endpoint."""

    features: Dict[str, float] = Field(
        ..., description="Feature names and numeric values matching feature_columns.json"
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log and count every request for monitoring."""
    path = request.url.path
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    if not path.startswith("/metrics"):
        REQUEST_COUNT.labels(request.method, path, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
    logger.info(
        "%s %s status=%s duration_ms=%.2f",
        request.method,
        path,
        response.status_code,
        elapsed * 1000,
    )
    return response


@app.get("/health")
def health():
    """Return 200 only when the saved model artifact can be loaded.

    Kubernetes readiness/liveness probes rely on the status code, so a pod with
    a missing or broken model must not report healthy.
    """
    try:
        load_model()
    except Exception as exc:  # any load failure means the pod is not ready
        logger.error("Model load failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "model_loaded": False, "detail": str(exc)},
        )
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    """Validate the request body and return a prediction with confidence."""
    try:
        result = predict(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    PREDICTION_COUNT.labels(result["label"]).inc()
    logger.info(
        "prediction=%s confidence=%.4f", result["prediction"], result["confidence"]
    )
    return result
