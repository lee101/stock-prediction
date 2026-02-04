#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

PAPER=1 BINANCE_TESTNET=1 BINANCE_BASE_ENDPOINT=https://testnet.binance.vision \
PYTHONUNBUFFERED=1 PYTHON_BIN="$(pwd)/.venv/bin/python" \
python -m binanceneural.trade_binance_hourly \
  --symbols SOLUSDT \
  --checkpoint binancechronossolexperiment2/checkpoints/chronos_sol_v2_test30_muon0975_20260204_114507/policy_checkpoint.pt \
  --horizon 1 \
  --sequence-length 72 \
  --forecast-horizons 1,4,24 \
  --forecast-cache-root binancechronossolexperiment/forecast_cache \
  --data-root trainingdatahourlybinance \
  --intensity-scale 1.0 \
  --min-gap-pct 0.0003 \
  --poll-seconds 30 \
  --expiry-minutes 90 \
  --price-tolerance 0.0008 \
  --dry-run
