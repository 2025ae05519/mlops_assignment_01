"""Prediction helper used by the API and quick local checks."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "heart_disease_pipeline.joblib"
FEATURES_PATH = ARTIFACTS / "feature_columns.json"


def load_model() -> tuple[Pipeline, list[str]]:
    """Load the trained sklearn pipeline and its expected feature order."""
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH), json.loads(FEATURES_PATH.read_text())


def predict(features: dict[str, float]) -> dict:
    """Run one prediction after checking that feature names match training."""
    model, columns = load_model()
    missing = [c for c in columns if c not in features]
    extra = [c for c in features if c not in columns]
    if missing or extra:
        raise ValueError(f"Feature mismatch; missing={missing}, unexpected={extra}")
    row = pd.DataFrame([[features[c] for c in columns]], columns=columns)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    positive_probability = float(probabilities[classes.index(1)])
    return {
        "prediction": prediction,
        "label": "disease_present" if prediction == 1 else "disease_not_detected",
        # Confidence is the probability the model assigns to its own prediction.
        "confidence": float(probabilities[classes.index(prediction)]),
        "positive_class_probability": positive_probability,
    }


if __name__ == "__main__":
    _, feature_columns = load_model()
    example = {column: 0.0 for column in feature_columns}
    print(json.dumps(predict(example), indent=2))
