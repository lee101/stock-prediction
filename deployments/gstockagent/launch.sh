#!/bin/bash
set -euo pipefail

cd /home/lee/code/stock
source .venv/bin/activate

export OPENPATHS_API_KEY="${OPENPATHS_API_KEY:-}"

# prod uses gemini-3.1-pro for quality, lite for backtesting
exec python -m gstockagent.trade_live \
    --model "${GSTOCK_MODEL:-gemini-3.1-pro}" \
    --leverage "${GSTOCK_LEVERAGE:-1.0}" \
    --max-positions "${GSTOCK_MAX_POS:-5}" \
    --interval-hours "${GSTOCK_INTERVAL:-24}" \
    --daemon \
    --live
