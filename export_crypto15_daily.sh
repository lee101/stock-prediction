#!/bin/bash
set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

SYMBOLS=BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,ADAUSD,UNIUSD,AAVEUSD,ALGOUSD,DOTUSD,SHIBUSD,XRPUSD,MATICUSD

echo "=== Exporting crypto15 daily TRAIN (up to 2025-06-01) ==="
python export_data_daily.py \
  --symbols "$SYMBOLS" \
  --data-root trainingdata \
  --output pufferlib_market/data/crypto15_daily_train.bin \
  --end-date 2025-06-01 \
  --union

echo ""
echo "=== Exporting crypto15 daily VAL (from 2025-06-01) ==="
python export_data_daily.py \
  --symbols "$SYMBOLS" \
  --data-root trainingdata \
  --output pufferlib_market/data/crypto15_daily_val.bin \
  --start-date 2025-06-01 \
  --min-days 30 \
  --union

echo ""
echo "=== Done ==="
ls -lh pufferlib_market/data/crypto15_daily_*.bin
