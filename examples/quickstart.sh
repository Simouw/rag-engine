#!/usr/bin/env bash
# Minimal end-to-end demo. Requires a configured .env with LLM settings.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "== readiness =="
curl -s "$BASE_URL/api/ready" | python -m json.tool

echo "== ingest sample document =="
python -m rag_engine ingest ./examples/docs --collection demo --init-db

echo "== search =="
curl -s "$BASE_URL/api/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"what does the light reaction produce?","top_k":3}' \
  | python -m json.tool

echo "== ask =="
curl -s "$BASE_URL/api/ask" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the light reaction produce?","top_k":3}' \
  | python -m json.tool
