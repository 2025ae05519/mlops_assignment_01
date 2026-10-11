# MLOps Assignment 01 — Heart Disease Classification

## Objective
Build a reproducible binary classifier for the UCI Heart Disease dataset, track experiments with MLflow, package inference behind a FastAPI service, test the service, and provide container/CI/deployment configuration.

## Dataset and target
The project uses UCI Heart Disease repository ID 45. Target values are mapped as `0 -> no disease` and values greater than `0 -> disease present`. The dataset is downloaded by `ucimlrepo` when a cached `data/raw/heart_disease_uci_raw.csv` is not present.

**Step-by-step execution guide (with the screenshot list): [docs/RUNBOOK.md](docs/RUNBOOK.md).** Report outline, architecture diagram and video plan: [docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md).

## Submission links
- Repository: https://github.com/2025ae05519/mlops_assignment_01
- Report: `report/` (add the final PDF/DOCX here)
- Video: _add link_
- API access: local only; see "Run API", "Docker" and "Kubernetes" below.

## Project structure
- `notebooks/heart_disease_experiment.ipynb`: exploration, plots, experimentation, and assignment record.
- `src/train.py`: data download and cleaning, train/test split, cross-validation, Random Forest tuning, model comparison, MLflow tracking (metrics, CSVs, ROC and confusion matrix plots), and model artifact generation.
- `src/predict.py`: loads the trained pipeline and produces a prediction with confidence (inference script).
- `src/api.py`: FastAPI `/health`, `/predict` and Prometheus `/metrics` endpoints with request logging.
- `tests/`: unit tests for data cleaning (`test_data.py`), model code (`test_model.py`) and the API (`test_api.py`).
- `data/raw/`, `data/processed/`: raw and cleaned dataset CSVs.
- `artifacts/`: generated model, feature contract, metrics, comparison CSVs and plots.
- `sample_request.json`, `scripts/smoke_test.sh`: sample input and a script that calls the API.
- `deployment/kubernetes.yaml`: Kubernetes Deployment and LoadBalancer Service; `deployment/ingress.yaml`: optional Ingress.
- `docker-compose.yml`, `monitoring/`: API + Prometheus + Grafana monitoring stack with a provisioned dashboard.
- `.github/workflows/ci.yml`: GitHub Actions workflow (lint, train, test, artifacts, Docker build and smoke test).
- `screenshots/`: evidence for the report.

## Environment setup
The Dockerfile and CI workflow use Python 3.12, so the dependency versions are pinned or bounded around that setup.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and track
Run from the repository root:

```bash
python -m src.train
```

This downloads or reads the dataset, saves cleaned data, compares Logistic Regression and tuned Random Forest, selects the model using training cross-validation ROC-AUC, evaluates the selected candidate on the held-out test set, logs model runs to MLflow using SQLite (`mlflow.db`), and writes files under `artifacts/`.

View MLflow locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Run tests
Training must be run once first so the model artifact exists:

```bash
python -m src.train
python -m pytest -v
```

## Run API
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- Health: `GET /health` (HTTP 503 if the model cannot be loaded)
- Predict: `POST /predict`
- Metrics: `GET /metrics/` (Prometheus format)
- Interactive docs: `/docs`

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
# {"prediction":0,"label":"disease_not_detected","confidence":0.62,"positive_class_probability":0.38}
```

The prediction request body is JSON with a `features` object containing every feature in `artifacts/feature_columns.json`. Feature names must match exactly. The API logs each request method, path, response status, and duration, and counts requests, latency and predictions in Prometheus metrics.

## Docker
Run training first so `artifacts/heart_disease_pipeline.joblib` and `artifacts/feature_columns.json` exist, then:

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
scripts/smoke_test.sh http://localhost:8000
```

## Monitoring
```bash
docker compose up --build
```
Prometheus runs on http://localhost:9090 and Grafana on http://localhost:3000 (admin / admin) with the "Heart Disease API" dashboard.

## Kubernetes
```bash
kubectl apply -f deployment/kubernetes.yaml
kubectl get pods,svc
```
The Service is a LoadBalancer: on Docker Desktop the API is at http://localhost/docs; on Minikube run `minikube image load heart-disease-api:latest` before applying and `minikube tunnel` to get an external IP. See [docs/RUNBOOK.md](docs/RUNBOOK.md) for the Ingress option. If the image is pushed to a remote registry, update the image name in the manifest.

## GitHub
Commit source code, notebook, tests, dependency file, deployment manifests, and model artifacts if allowed by your assignment/repository policy. Do not commit secrets, virtual environments, the MLflow database, or large MLflow artifact directories.

## Results and evidence
Current selected model from `artifacts/final_metrics.json`: Logistic Regression.

- Accuracy: 0.8689
- Precision: 0.8125
- Recall: 0.9286
- F1-score: 0.8667
- ROC-AUC: 0.9502

Screenshots for CI, MLflow, Docker, Kubernetes and monitoring go in `screenshots/` (the list is in [docs/RUNBOOK.md](docs/RUNBOOK.md)).
