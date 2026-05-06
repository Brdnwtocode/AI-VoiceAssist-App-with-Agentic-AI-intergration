#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x "venv/bin/python" ]]; then
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate
pip install -r requirements.txt -q

export MOCK_OPENAI=1
export SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:8000}"

python -m uvicorn main:app --host 127.0.0.1 --port 8000 &
UV_PID=$!
cleanup() { kill "$UV_PID" 2>/dev/null || true; }
trap cleanup EXIT
sleep 3

python test_contract.py --mock-openai
