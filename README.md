# MLOps Assignment 01 — Heart Disease Classification

## Objective
Build a reproducible binary classifier for the UCI Heart Disease dataset, track experiments with MLflow, package inference behind a FastAPI service, test the service, and provide container/CI/deployment configuration...

## Dataset and target
The project uses UCI Heart Disease repository ID 45. Target values are mapped as `0 -> no disease` and values greater than `0 -> disease present`. The dataset is downloaded by `ucimlrepo` when a cached `data/raw/heart_disease_uci_raw.csv` is not present.

**Step-by-step execution guide: see [Running the full flow on BITS Prayog Shala](#running-the-full-flow-on-bits-prayog-shala) below.** Architecture diagram: [report/architecture_diagram.png](report/architecture_diagram.png). Report outline and video plan: [docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md).

## Submission links
- Repository: https://github.com/2025ae05519/mlops_assignment_01
- Report: [report/MLOps_Assignment_01_Report.docx](report/MLOps_Assignment_01_Report.docx)
- Video: _add link_
- API access: not publicly hosted; run it locally or on Minikube by following the Prayog Shala steps below.

## Project structure
- `notebooks/heart_disease_experiment.ipynb`: exploration, plots, experimentation, and assignment record.
- `src/train.py`: data download and cleaning, preprocessing pipelines (median imputation, scaling for Logistic Regression, one-hot encoding of `cp`, `restecg`, `slope`, `thal`), train/test split, cross-validation, Random Forest tuning, model comparison, MLflow tracking (metrics, CSVs, ROC and confusion matrix plots), and model artifact generation.
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

## Running the full flow on BITS Prayog Shala
This is the exact sequence used to produce the results and screenshots in `screenshots/`. Run it in the Prayog Shala web-RDP desktop (VS Code terminal, Python 3.12, Docker and Minikube are available there). Each 📸 marks the screenshot file it produces. Crop the browser address bar out of every screenshot, because the web-RDP URL contains a session token.

### 0. Open the project and get the latest code
```bash
cd ~/workspace/MLOPS_Assignment_01/mlops_assignment_01
source .venv/bin/activate          # first time: python -m venv .venv && source .venv/bin/activate
git status                         # stash or commit local changes first if any are listed
git pull origin main
python -m pip install -r requirements.txt
python --version && git log --oneline -3
```
📸 `01_project_environment.png`: project path, Python version, branch and recent commits.

### 1. Train the models (downloads the data on the first run)
```bash
python -m src.train
ls -lh data/raw data/processed artifacts artifacts/plots
cat artifacts/final_metrics.json
```
Expected: `Selected by CV ROC-AUC: logistic_regression`, four PNGs in `artifacts/plots/`, and in `final_metrics.json` a test `roc_auc` of about 0.958 and `cv_roc_auc_mean` of about 0.907. The raw CSV is cached in `data/raw/`, so later runs work offline.

📸 `02_training_output.png`: end of the training output with the classification report.
📸 `03_artifacts_and_dataset.png`: the `ls -lh` listing and `final_metrics.json`.

### 2. Lint and unit tests
```bash
ruff check --select F src tests
python -m pytest -v
```
Expected: `All checks passed!` and **16 passed** (5 API, 5 data, 6 model tests). The Starlette and pandas warnings are harmless.

📸 `04_lint_and_tests.png`

### 3. MLflow UI (keep this terminal open)
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
Open port 5000 in the lab browser, experiment **heart-disease-classification**. The newest two runs are `logistic_regression` and `random_forest_tuned`.

📸 `05_mlflow_runs.png`: runs table with the `cv_roc_auc_mean`, `test_accuracy`, `test_recall` and `test_roc_auc` columns.
📸 `06_mlflow_compare.png`: both runs ticked, **Compare**.
📸 `07_mlflow_params_metrics.png`: the `random_forest_tuned` run with its parameters and metrics.
📸 `08_mlflow_plots.png`: **Artifacts → plots →** ROC curve.
📸 `09_mlflow_model.png`: **Models** list (or **Artifacts → model**).

### 4. EDA notebook
```bash
python -m jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser   # pip install notebook if missing
```
Open `notebooks/heart_disease_experiment.ipynb` and run all cells (or view the saved outputs).

📸 `10_eda_class_balance.png`, `11_eda_histograms.png`, `12_eda_correlation.png`

### 5. Docker: build and test the container
```bash
docker build -t heart-disease-api:latest .
docker images heart-disease-api
docker run -d --rm --name heart-api-test -p 8001:8000 heart-disease-api:latest
sleep 30
docker ps --filter name=heart-api-test      # STATUS shows (healthy)
scripts/smoke_test.sh http://127.0.0.1:8001 3
docker logs heart-api-test --tail 10
docker stop heart-api-test
```
Expected `/predict` response for `sample_request.json`: `{"prediction":0,"label":"disease_not_detected","confidence":0.82,"positive_class_probability":0.18}`.

📸 `13_docker_build.png`: end of the build and `docker images`.
📸 `14_docker_run_test.png`: `(healthy)` status, smoke test output and container logs.

### 6. Deploy to Minikube
```bash
minikube status
minikube image load heart-disease-api:latest
minikube kubectl -- apply -f deployment/kubernetes.yaml
minikube kubectl -- rollout restart deployment/heart-disease-api    # same "latest" tag, so force new pods
minikube kubectl -- rollout status deployment/heart-disease-api
minikube kubectl -- get deployments,pods,services -o wide
```
Expected: deployment **2/2** ready, two pods `Running`, Service type `LoadBalancer`.

📸 `15_k8s_resources.png`
📸 `16_k8s_rollout.png`: "successfully rolled out".

### 7. Call the API through Kubernetes
Get a URL with one of these (in a separate terminal):
```bash
minikube tunnel                                                  # then use the Service EXTERNAL-IP
minikube service heart-disease-api --url                         # or this URL
minikube kubectl -- port-forward service/heart-disease-api 8000:80   # fallback: http://127.0.0.1:8000
```
Then:
```bash
BASE=http://127.0.0.1:8000          # replace with your URL
curl -i $BASE/health
scripts/smoke_test.sh $BASE 20
```
Open `$BASE/docs`, expand **POST /predict**, **Try it out**, paste `sample_request.json`, **Execute**.

📸 `17_api_health.png`: `/health` returns 200 with `model_loaded: true`.
📸 `18_api_predict.png`: Swagger `/predict` response with `confidence`.

### 8. Logs, probes and resources
```bash
minikube kubectl -- logs deployment/heart-disease-api --tail=30
minikube kubectl -- describe deployment heart-disease-api | sed -n '1,45p'
```
📸 `19_app_logs.png`: the request-logging middleware output (`GET /health status=200 duration_ms=...`) for every readiness and liveness probe.
📸 `20_probes_and_resources.png`: replicas, liveness/readiness probes on `/health`, CPU/memory limits.

### 9. Prometheus and Grafana
Stop any port-forward on port 8000 first.
```bash
docker compose up -d --build
docker compose ps
```
- Prometheus (port 9090): **Status → Targets** shows `heart-disease-api` UP; in **Graph** run `sum by (path) (rate(heart_api_requests_total[1m]))`.
- Grafana (port 3000, `admin` / `admin`): **Dashboards → Heart Disease API**. The panels show requests per second and p95 latency by path.

📸 `21_prometheus_targets.png`: the `heart-disease-api` target is UP.
📸 `22_prometheus_query.png`: request rate by path from the query above.
📸 `23_grafana_dashboard.png`: requests per second and p95 latency panels.

```bash
docker compose down
```

### 10. GitHub Actions
Push your changes, then open the repository's **Actions** tab and the latest `mlops-ci` run.

📸 `24_ci_pipeline.png`: the workflow graph, where "Build and smoke-test Docker image" waits for "Lint, train and test".
📸 `25_ci_test_steps.png`: the "Lint, train and test" job steps with 16 tests passing.
📸 `26_ci_docker_steps.png`: the green Docker job ("Run container", "Wait for /health", "Call /predict with sample input", "Check /metrics").
📸 `27_pr_checks.png`: the workflow runs list (or a pull request showing "All checks have passed").

### Troubleshooting
- `ModuleNotFoundError: prometheus_client`: rerun `pip install -r requirements.txt`.
- Pods not Ready or `CrashLoopBackOff`: check `minikube kubectl -- logs <pod>`; `/health` returns 503 when the model can't load. Retrain, rebuild the image and run `minikube image load` again.
- Old response (confidence 0.62): Minikube is still running the old image; repeat `minikube image load` and `rollout restart`.
- Port 8000 in use: stop the port-forward or run `docker compose down`.

## Quick reference (any machine)

### Environment setup
The Dockerfile and CI workflow use Python 3.12, so the dependency versions are pinned or bounded around that setup.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Train and track
Run from the repository root:

```bash
python -m src.train
```

This downloads or reads the dataset, saves cleaned data, compares Logistic Regression and tuned Random Forest, selects the model using training cross-validation ROC-AUC, evaluates the selected candidate on the held-out test set, logs model runs to MLflow using SQLite (`mlflow.db`), and writes files under `artifacts/`.

View MLflow locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

### Run tests
Training must be run once first so the model artifact exists:

```bash
python -m src.train
python -m pytest -v
```

### Run API
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- Health: `GET /health` (HTTP 503 if the model cannot be loaded)
- Predict: `POST /predict`
- Metrics: `GET /metrics/` (Prometheus format)
- Interactive docs: `/docs`

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
# {"prediction":0,"label":"disease_not_detected","confidence":0.82,"positive_class_probability":0.18}
```

The prediction request body is JSON with a `features` object containing every feature in `artifacts/feature_columns.json`. Feature names must match exactly. The API logs each request method, path, response status, and duration, and counts requests, latency and predictions in Prometheus metrics.

### Docker
Run training first so `artifacts/heart_disease_pipeline.joblib` and `artifacts/feature_columns.json` exist, then:

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
scripts/smoke_test.sh http://localhost:8000
```

### Monitoring
```bash
docker compose up --build
```
Prometheus runs on http://localhost:9090 and Grafana on http://localhost:3000 (admin / admin) with the "Heart Disease API" dashboard.

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes.yaml
kubectl get pods,svc
```
The Service is a LoadBalancer: on Docker Desktop the API is at http://localhost/docs; on Minikube run `minikube image load heart-disease-api:latest` before applying and `minikube tunnel` to get an external IP. For the optional Ingress, see the comments in `deployment/ingress.yaml`. If the image is pushed to a remote registry, update the image name in the manifest.

## GitHub
Commit source code, notebook, tests, dependency file, deployment manifests, and model artifacts if allowed by your assignment/repository policy. Do not commit secrets, virtual environments, the MLflow database, or large MLflow artifact directories.

## Results and evidence
Current selected model from `artifacts/final_metrics.json`: Logistic Regression.

- Accuracy: 0.8689
- Precision: 0.8125
- Recall: 0.9286
- F1-score: 0.8667
- ROC-AUC: 0.9578

Screenshots for every step are in `screenshots/`; the Prayog Shala section above says which step produces each file.
