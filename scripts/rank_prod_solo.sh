#!/usr/bin/env bash
set -u
OUT=artifacts/gate_sweep/prod_solo_c0.txt
mkdir -p artifacts/gate_sweep
: > "$OUT"
for p in pufferlib_market/prod_ensemble/*.pt; do
  name=$(basename "$p" .pt)
  line=$(.venv313/bin/python trade_daily_stock_prod.py \
    --backtest --paper --backtest-days 120 --backtest-starting-cash 10000 \
    --backtest-entry-offset-bps 5 --backtest-exit-offset-bps 25 \
    --allocation-pct 12.5 --min-open-confidence 0 --min-open-value-estimate 0 \
    --checkpoint "$p" --no-ensemble \
    --data-source local --data-dir trainingdata \
    --execution-backend alpaca 2>&1 | grep 'Backtest results' | sed 's/.*Backtest results: //')
  printf '%s\t%s\n' "$name" "$line" | tee -a "$OUT"
done
