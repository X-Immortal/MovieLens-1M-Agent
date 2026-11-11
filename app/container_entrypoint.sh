#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${RUNTIME_DIR:-/runtime}/runs"
exec streamlit run /app/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true \
  --browser.gatherUsageStats=false
