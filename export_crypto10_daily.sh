#!/bin/bash
set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

# Train split (up to 2025-06-01)
python export_data_daily.py \
  --symbols BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,ADAUSD,UNIUSD,AAVEUSD \
  --data-root trainingdata \
  --output pufferlib_market/data/crypto10_daily_train.bin \
  --end-date 2025-06-01

# Val split (2025-06-01 onwards)
python export_data_daily.py \
  --symbols BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,ADAUSD,UNIUSD,AAVEUSD \
  --data-root trainingdata \
  --output pufferlib_market/data/crypto10_daily_val.bin \
  --start-date 2025-06-01 \
  --min-days 30
