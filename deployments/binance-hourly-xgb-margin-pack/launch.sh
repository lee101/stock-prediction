#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lee/code/stock"
PYTHON_BIN="$ROOT/.venv313/bin/python"

cd "$ROOT"

set +u
source /home/lee/.secretbashrc 2>/dev/null || true
set -u

export ALLOW_BINANCE_XGB_LIVE_TRADING=1
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" "$ROOT/scripts/binance_hourly_xgb_margin_trader.py" \
  --execute \
  --daemon \
  --run-on-start \
  --cycle-minutes 60 \
  --refresh-data-before-cycle \
  --json-out "$ROOT/analysis/binance_hourly_xgb_margin_plan_latest.json"
