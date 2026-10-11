# Runbook: executing the full pipeline

Follow these steps in order on your own machine. Each step lists the screenshot to take for the report (save them in `screenshots/` with the suggested file name).

Prerequisites: Python 3.12, Git, Docker Desktop (with Kubernetes enabled) **or** Docker + Minikube, and `kubectl`.

## 1. Clone and set up the environment

```bash
git clone https://github.com/2025ae05519/mlops_assignment_01.git
cd mlops_assignment_01
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Data acquisition and EDA

The dataset is downloaded from the UCI repository (id 45) by `ucimlrepo`. The raw copy is saved to `data/raw/heart_disease_uci_raw.csv` and the cleaned copy to `data/processed/heart_disease_clean.csv` (both are committed).

```bash
jupyter notebook notebooks/heart_disease_experiment.ipynb   # pip install notebook if needed
```

Run all cells top to bottom.

- 📸 `01_eda_class_balance.png`, `02_eda_histograms.png`, `03_eda_correlation.png`: the EDA plots in notebook section 5.

## 3. Train models and track experiments

```bash
python -m src.train
```

This compares Logistic Regression and a tuned Random Forest with 5-fold CV, logs both runs to MLflow (params, metrics, CSVs, confusion matrix and ROC plots, model), and saves the selected model to `artifacts/`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then commit the dataset CSVs and the refreshed artifacts (the cleaned dataset is a required deliverable):

```bash
git add data/raw/heart_disease_uci_raw.csv data/processed/heart_disease_clean.csv artifacts/
git commit -m "Add raw and cleaned dataset and training artifacts"
git push
```

Open http://127.0.0.1:5000, experiment `heart-disease-classification`.

- 📸 `04_mlflow_runs.png`: the runs list with metrics columns.
- 📸 `05_mlflow_run_detail.png`: one run's params and metrics.
- 📸 `06_mlflow_plots.png`: the run's Artifacts tab showing `plots/` (ROC curve, confusion matrix).

## 4. Unit tests and lint

```bash
ruff check --select F src tests
python -m pytest -v
```

- 📸 `07_pytest_pass.png`: terminal with all tests passing.

## 5. Run the API locally

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
scripts/smoke_test.sh http://localhost:8000 3
# or manually:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Swagger UI: http://localhost:8000/docs.

- 📸 `08_api_docs.png`: Swagger UI with a `/predict` call and its response (prediction + confidence).

## 6. Docker container

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
# second terminal
scripts/smoke_test.sh http://localhost:8000 3
```

- 📸 `09_docker_build.png`: successful build output.
- 📸 `10_docker_run.png`: container logs showing the logged requests next to the curl output.

Stop the container (Ctrl+C) before the next step so port 8000 is free.

## 7. Kubernetes deployment

### Option A: Docker Desktop Kubernetes

The image built in step 6 is already visible to the cluster.

```bash
kubectl apply -f deployment/kubernetes.yaml
kubectl get pods,svc
scripts/smoke_test.sh http://localhost:80 3
```

### Option B: Minikube

```bash
minikube start
minikube image load heart-disease-api:latest
kubectl apply -f deployment/kubernetes.yaml
minikube tunnel                     # keep running in its own terminal
kubectl get svc heart-disease-api   # EXTERNAL-IP is now set
scripts/smoke_test.sh http://<EXTERNAL-IP> 3
```

Optional Ingress on Minikube: `minikube addons enable ingress`, `kubectl apply -f deployment/ingress.yaml`, add `<minikube ip> heart-api.local` to your hosts file, then open http://heart-api.local/docs.

- 📸 `11_k8s_pods_svc.png`: `kubectl get pods,svc` showing 2 running pods and the LoadBalancer service.
- 📸 `12_k8s_predict.png`: the smoke test against the cluster URL.
- 📸 `13_k8s_logs.png`: `kubectl logs deploy/heart-disease-api` showing request logs.

Clean up with `kubectl delete -f deployment/kubernetes.yaml`.

## 8. Monitoring (Prometheus + Grafana)

```bash
docker compose up --build
scripts/smoke_test.sh http://localhost:8000 50   # generate traffic
```

- Raw metrics: http://localhost:8000/metrics/
- Prometheus: http://localhost:9090 (try the query `heart_api_requests_total`)
- Grafana: http://localhost:3000 (admin / admin), dashboard **Heart Disease API**

- 📸 `14_metrics_endpoint.png`, `15_prometheus_query.png`, `16_grafana_dashboard.png`.

Stop with `docker compose down`.

## 9. CI/CD

Every push runs `.github/workflows/ci.yml`:

1. **Lint, train and test**: ruff, `python -m src.train`, pytest (JUnit report), then uploads `artifacts/`, data, the training log and test results as a run artifact.
2. **Build and smoke-test Docker image**: builds the image, starts it, waits for `/health`, calls `/predict` with `sample_request.json`, checks `/metrics`, and prints the container logs.

Any lint error, failing test, failed training run or unhealthy container fails the pipeline.

- 📸 `17_ci_success.png`: the Actions run summary with both jobs green.
- 📸 `18_ci_artifacts.png`: the uploaded artifact at the bottom of the run page.
- 📸 `19_ci_failure.png` (optional but strong evidence): push a deliberately failing test on a branch, screenshot the red run and log, then revert.
