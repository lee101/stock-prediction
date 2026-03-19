#!/usr/bin/env bash
set -euo pipefail

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

current_pid="${1:-2205592}"

while kill -0 "$current_pid" >/dev/null 2>&1; do
  sleep 30
done

python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols PFE \
  --base-stock-universe live20260318 \
  --base-long-only-symbols NVDA,PLTR,GOOG,AAPL,MSFT,META,TSLA,NET,DBX,SOFI,INTC,MU,TTD,PATH,NBIS,TME \
  --output-dir analysis/alpaca_stock_expansion_pfe_20260318
