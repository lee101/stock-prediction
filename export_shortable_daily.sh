#!/bin/bash
set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

python export_data_daily.py \
  --symbols YELP,EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA \
  --data-root trainingdata/train \
  --output pufferlib_market/data/shortable10_daily_train.bin \
  --end-date 2025-06-01

python export_data_daily.py \
  --symbols YELP,EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA \
  --data-root trainingdata/train \
  --output pufferlib_market/data/shortable10_daily_val.bin \
  --start-date 2025-06-01 --min-days 30
