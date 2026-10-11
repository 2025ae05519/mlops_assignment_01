#!/usr/bin/env bash
# Send health and prediction requests to a running API.
# Usage: scripts/smoke_test.sh [base_url] [count]
set -euo pipefail
BASE_URL="${1:-http://localhost:8000}"
COUNT="${2:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

curl -fsS "$BASE_URL/health"; echo
for _ in $(seq "$COUNT"); do
  curl -fsS -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d @"$ROOT/sample_request.json"
  echo
done
