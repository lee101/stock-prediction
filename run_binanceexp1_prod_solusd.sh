#!/usr/bin/env bash
set -euo pipefail

source .venv313/bin/activate

python -m binanceexp1.trade_binance_hourly \
  --symbols SOLUSD \
  --checkpoint binanceneural/checkpoints/solusd_h1_ft_20260203/epoch_005.pt \
  --horizon 1 \
  --sequence-length 96 \
  --intensity-scale 1.4 \
  --min-gap-pct 0.0003 \
  --poll-seconds 30 \
  --expiry-minutes 90 \
  --cycle-minutes 5 \
  --log-metrics \
  --metrics-log-path strategy_state/binanceexp1-solusd-metrics.csv
