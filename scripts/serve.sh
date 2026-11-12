#!/usr/bin/env bash
# Serve the API from the GPU env (.venv-ocr) on 127.0.0.1:18300; logs go to data/serve.log.
#   CORS_ORIGINS=https://my-app.vercel.app scripts/serve.sh      (comma-separated; default the Vite dev server)
# Background it with: setsid nohup scripts/serve.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:18373}"
mkdir -p data
exec .venv-ocr/bin/python -s -m uvicorn api.main:app --host 127.0.0.1 --port 18300 >> data/serve.log 2>&1
