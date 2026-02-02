#!/usr/bin/env bash
set -euo pipefail

source .venv313/bin/activate

python -m binanceexp1.trade_binance_hourly \
  --symbols SOLUSD \
  --checkpoint binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt \
  --horizon 24 \
  --sequence-length 96 \
  --intensity-scale 20.0 \
  --min-gap-pct 0.0003 \
  --poll-seconds 30 \
  --expiry-minutes 90
