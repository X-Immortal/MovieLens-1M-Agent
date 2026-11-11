#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${RUNTIME_DIR:-/runtime}/runs" "${RUNTIME_DIR:-/runtime}/sessions"
exec python3 /app/web_server.py
