#!/bin/bash
set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

python export_data_daily.py \
  --symbols AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,PLTR,NET,NFLX,AMD,ADBE,CRM,PYPL,INTC \
  --data-root trainingdata \
  --output pufferlib_market/data/stocks15_daily_train.bin \
  --end-date 2025-06-01

python export_data_daily.py \
  --symbols AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,PLTR,NET,NFLX,AMD,ADBE,CRM,PYPL,INTC \
  --data-root trainingdata \
  --output pufferlib_market/data/stocks15_daily_val.bin \
  --start-date 2025-06-01 --min-days 30
