#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 scripts/init_data.py
mkdir -p runtime
if [[ ! -e .env ]]; then
  cp .env.example .env
fi
docker compose build
docker compose up -d
echo "MovieLens Agent: http://localhost:8501"
echo "Logs:      docker compose logs -f movielens-agent"
