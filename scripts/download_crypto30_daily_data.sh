#!/bin/bash
# Aggregate existing binance_spot_hourly/ data to daily CSVs for crypto30 PPO training.
# Most hourly data already downloaded; this just does hourly -> daily aggregation.
#
# Usage:
#   bash scripts/download_crypto30_daily_data.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

HOURLY_ROOT="binance_spot_hourly"
OUTPUT_DIR="trainingdata/train"

SYMBOLS=(
    BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AVAXUSDT LINKUSDT AAVEUSDT LTCUSDT
    XRPUSDT DOTUSDT UNIUSDT NEARUSDT APTUSDT ICPUSDT SHIBUSDT ADAUSDT
    FILUSDT ARBUSDT OPUSDT INJUSDT SUIUSDT TIAUSDT SEIUSDT ATOMUSDT
    ALGOUSDT BCHUSDT BNBUSDT TRXUSDT PEPEUSDT MATICUSDT
)

mkdir -p "$OUTPUT_DIR"

echo "=== Aggregating ${#SYMBOLS[@]} crypto pairs: hourly -> daily ==="
echo "  hourly_root=$HOURLY_ROOT  output=$OUTPUT_DIR"

# First refresh hourly data from Binance Vision (fill any gaps)
echo "--- Refreshing hourly data from Binance Vision ---"
python scripts/collect_binance_vision_klines.py \
    --symbols ${SYMBOLS[@]} \
    --interval 1h \
    --out-root "$HOURLY_ROOT" \
    --daily-lookback-days 30

# Aggregate hourly -> daily
echo "--- Aggregating to daily bars ---"
python scripts/build_binance_spot_daily_from_hourly.py \
    --symbols ${SYMBOLS[@]} \
    --hourly-root "$HOURLY_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --expected-bars-per-day 24 \
    --force

echo "=== Done. Daily CSVs in $OUTPUT_DIR ==="
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l
echo "CSV files written"
