"""Train the heart disease models and record the experiment in MLflow.

Run from the repository root:
    python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend for CI and Docker
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
for folder in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, PLOTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
EXPERIMENT_NAME = "heart-disease-classification"


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the UCI data, clean the target, and save a processed CSV copy."""
    raw_path = RAW_DIR / "heart_disease_uci_raw.csv"
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
    else:
        heart = fetch_ucirepo(id=45)
        features = heart.data.features.copy()
        targets = heart.data.targets.copy()
        target_col = targets.columns[0]
        raw_df = features.copy()
        raw_df[target_col] = targets[target_col].values
        raw_df.to_csv(raw_path, index=False)

    df = clean_dataframe(raw_df)
    df.to_csv(PROCESSED_DIR / "heart_disease_clean.csv", index=False)
    return split_features_target(df)


def clean_dataframe(raw_df: pd.DataFrame, target_col: str = "num") -> pd.DataFrame:
    """Convert missing markers to NaN, coerce numbers, drop rows without a
    target, and map the 0-4 target to binary (0 = no disease, 1 = disease)."""
    if target_col not in raw_df.columns:
        target_col = raw_df.columns[-1]
    df = raw_df.replace(["?", "NA", "N/A", ""], np.nan).copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = (df[target_col].astype(float) > 0).astype(int)
    if target_col != "num":
        df = df.rename(columns={target_col: "num"})
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split a cleaned frame into features and target, dropping empty columns."""
    X = df.drop(columns=["num"])
    y = df["num"].astype(int)
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)
    return X, y, X.columns.tolist()


def save_evaluation_plots(name: str, model: Pipeline, X_test, y_test) -> list[Path]:
    """Save confusion matrix and ROC curve PNGs for one model."""
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {name}")
    fig.tight_layout()
    paths.append(PLOTS_DIR / f"{name}_confusion_matrix.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title(f"ROC curve: {name}")
    fig.tight_layout()
    paths.append(PLOTS_DIR / f"{name}_roc_curve.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def build_models(features: list[str]) -> tuple[Pipeline, Pipeline]:
    """Create the two sklearn pipelines used for comparison."""
    scaled = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), features)
    ])
    unscaled = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), features)
    ])
    lr = Pipeline([
        ("preprocessor", scaled),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    rf = Pipeline([
        ("preprocessor", unscaled),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            class_weight="balanced", min_samples_leaf=2
        )),
    ])
    return lr, rf


def main() -> None:
    """Run the full training, evaluation, tracking, and artifact save flow."""
    X, y, features = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall",
        "f1": "f1", "roc_auc": "roc_auc",
    }
    lr, rf = build_models(features)

    cv_results = {}
    for name, model in (("logistic_regression", lr), ("random_forest", rf)):
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=False, error_score="raise",
        )
        cv_results[name] = {
            metric: float(np.mean(scores[f"test_{metric}"])) for metric in scoring
        }
    cv_df = pd.DataFrame(cv_results).T.sort_values("roc_auc", ascending=False)

    search = GridSearchCV(
        estimator=rf,
        param_grid={
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    best_lr = lr.fit(X_train, y_train)

    candidates = {
        "logistic_regression": best_lr,
        "random_forest_tuned": best_rf,
    }
    # Choose by cross-validation performance, not by held-out test performance.
    cv_auc_by_candidate = {
        "logistic_regression": cv_df.loc["logistic_regression", "roc_auc"],
        "random_forest_tuned": search.best_score_,
    }
    best_name = max(cv_auc_by_candidate, key=cv_auc_by_candidate.get)
    best_model = candidates[best_name]

    rows = []
    for name, model in candidates.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "cv_roc_auc_mean": float(cv_auc_by_candidate[name]),
        })
    evaluation_df = pd.DataFrame(rows)
    cv_df.to_csv(ARTIFACTS_DIR / "cross_validation_results.csv")
    evaluation_df.to_csv(ARTIFACTS_DIR / "test_evaluation_results.csv", index=False)

    # SQLite-backed tracking database and artifact store remain in project root.
    mlflow_db = ROOT / "mlflow.db"
    mlflow_artifacts = ROOT / "mlartifacts"
    mlflow_artifacts.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=mlflow_artifacts.resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    for name, model in candidates.items():
        row = evaluation_df.loc[evaluation_df["model"] == name].iloc[0]
        sample_input = X_train.head(3).astype("float64")
        signature = infer_signature(sample_input, model.predict(sample_input))
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("assignment", "MLOps Assignment 01")
            mlflow.set_tag("dataset", "UCI Heart Disease, repository ID 45")
            mlflow.set_tag("selected_model", str(name == best_name))
            mlflow.log_params({
                "model_name": name, "dataset_id": 45,
                "target_mapping": "0 -> 0; values > 0 -> 1",
                "random_state": RANDOM_STATE, "test_size": 0.2,
                "cross_validation_folds": 5,
            })
            if name == "random_forest_tuned":
                mlflow.log_params(search.best_params_)
            for metric in ("cv_roc_auc_mean", "accuracy", "precision", "recall", "f1", "roc_auc"):
                mlflow.log_metric(
                    metric if metric == "cv_roc_auc_mean" else f"test_{metric}",
                    float(row[metric]),
                )
            mlflow.log_artifact(str(ARTIFACTS_DIR / "cross_validation_results.csv"), artifact_path="evaluation")
            mlflow.log_artifact(str(ARTIFACTS_DIR / "test_evaluation_results.csv"), artifact_path="evaluation")
            for plot_path in save_evaluation_plots(name, model, X_test, y_test):
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
            mlflow.sklearn.log_model(
                sk_model=model, name="model", input_example=sample_input,
                signature=signature, serialization_format="pickle",
            )
            print(f"MLflow run: {name}, run_id={run.info.run_id}")

    joblib.dump(best_model, ARTIFACTS_DIR / "heart_disease_pipeline.joblib")
    (ARTIFACTS_DIR / "feature_columns.json").write_text(json.dumps(features, indent=2))
    final_row = evaluation_df.loc[evaluation_df["model"] == best_name].iloc[0].to_dict()
    final_row["selected_model"] = best_name
    final_row["feature_columns"] = features
    final_row["best_random_forest_params"] = search.best_params_
    (ARTIFACTS_DIR / "final_metrics.json").write_text(json.dumps(final_row, indent=2, default=float))
    print("\nSelected by CV ROC-AUC:", best_name)
    print("Held-out test report:")
    print(classification_report(y_test, best_model.predict(X_test), zero_division=0))
    print("Artifacts saved to:", ARTIFACTS_DIR)
    print("MLflow tracking DB:", mlflow_db)


if __name__ == "__main__":
    main()
