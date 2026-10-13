"""Unit tests for model building, plotting and prediction helpers."""
import numpy as np
import pandas as pd
import pytest

from src import predict as predict_module
from src.train import build_models, save_evaluation_plots


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    X.iloc[0, 1] = np.nan  # pipelines must impute missing values
    y = pd.Series((X["a"].fillna(0) + X["c"] > 0).astype(int))
    return X, y


def test_both_pipelines_fit_and_return_probabilities(synthetic_data):
    X, y = synthetic_data
    for model in build_models(list(X.columns)):
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_pipeline_scales_and_random_forest_does_not(synthetic_data):
    X, _ = synthetic_data
    lr, rf = build_models(list(X.columns))
    lr_steps = lr.named_steps["preprocessor"].transformers[0][1].named_steps
    rf_steps = rf.named_steps["preprocessor"].transformers[0][1].named_steps
    assert "scaler" in lr_steps
    assert "scaler" not in rf_steps


def test_evaluation_plots_are_written(synthetic_data, tmp_path, monkeypatch):
    X, y = synthetic_data
    monkeypatch.setattr("src.train.PLOTS_DIR", tmp_path)
    lr, _ = build_models(list(X.columns))
    lr.fit(X, y)
    paths = save_evaluation_plots("lr", lr, X, y)
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_predict_rejects_wrong_feature_names():
    _, columns = predict_module.load_model()
    features = {c: 0.0 for c in columns[:-1]}
    features["not_a_feature"] = 1.0
    with pytest.raises(ValueError, match="Feature mismatch"):
        predict_module.predict(features)


def test_predict_confidence_matches_predicted_class():
    _, columns = predict_module.load_model()
    result = predict_module.predict({c: 1.0 for c in columns})
    p = result["positive_class_probability"]
    expected = p if result["prediction"] == 1 else 1 - p
    assert result["confidence"] == pytest.approx(expected)
    assert result["confidence"] >= 0.5


def test_nominal_features_are_one_hot_encoded():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({
        "age": rng.integers(30, 80, size=60).astype(float),
        "cp": rng.integers(1, 5, size=60).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], size=60),
    })
    X.loc[0, "thal"] = np.nan  # categorical missing values are imputed too
    y = pd.Series(rng.integers(0, 2, size=60))
    for model in build_models(list(X.columns)):
        model.fit(X, y)
        preprocessor = model.named_steps["preprocessor"]
        assert preprocessor.transformers_[1][2] == ["cp", "thal"]
        # 1 numeric column + 4 chest pain codes + 3 thal codes
        assert preprocessor.transform(X).shape[1] == 8
        # an unseen category must not break prediction
        unseen = X.head(1).assign(cp=9.0)
        assert model.predict_proba(unseen).shape == (1, 2)
