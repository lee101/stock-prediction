#!/bin/bash
# Efficient retrain sweep - BF16 split AMP + vectorized sim
# ~2.77x faster than FP32 baseline
set -e
cd /nvme0n1-disk/code/stock-prediction
export PYTHONPATH=/nvme0n1-disk/code/stock-prediction

VENV=.venv312/bin/python
TRAIN="trainingefficiency/train_efficient.py"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

COMMON="--no-compile --hidden-dim 512 --num-layers 6 --num-heads 8 \
  --model-arch classic --maker-fee 0.001 --margin-annual-rate 0.0625 \
  --max-leverage 2.0 --fill-buffer-pct 0.0005 --epochs 20 --batch-size 64 \
  --lr 1e-5 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 \
  --loss-type sortino --forecast-horizons 1,24"

echo "=== EFFICIENT STOCK RETRAIN ==="

echo "[STOCK] wd=0.04 rw=0.15 seq=48 (bf16 split + vsim)"
$VENV $TRAIN \
  --symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT \
  --sequence-length 48 --weight-decay 0.04 --return-weight 0.15 --seed 1337 \
  --checkpoint-name efficient_stock_wd04_rw15 $COMMON

echo ""
echo "=== EFFICIENT ETH RETRAIN ==="

echo "[ETH] wd=0.04 rw=0.15 seq=48 (bf16 split + vsim)"
$VENV $TRAIN \
  --symbols ETHUSD \
  --stock-data-root trainingdatahourly/crypto \
  --stock-cache-root binanceneural/forecast_cache \
  --sequence-length 48 --weight-decay 0.04 --return-weight 0.15 --seed 1337 \
  --checkpoint-name efficient_eth_wd04_rw15 $COMMON

echo ""
echo "=== EVALS ==="

for cfg in efficient_stock_wd04_rw15; do
  echo "[EVAL STOCK] $cfg"
  $VENV $SWEEP \
    --checkpoint-dir unified_hourly_experiment/checkpoints/$cfg \
    --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT \
    --holdout-days 3,7,14,30 \
    --fee-rate 0.001 --margin-rate 0.0625 --no-close-at-eod \
    --max-hold-hours 6 --max-positions 7 --leverage 2.0 --min-edge 0.0 \
    2>&1 | tail -30
  echo ""
done

for cfg in efficient_eth_wd04_rw15; do
  echo "[EVAL ETH] $cfg"
  $VENV $SWEEP \
    --checkpoint-dir unified_hourly_experiment/checkpoints/$cfg \
    --symbols ETHUSD \
    --data-root trainingdatahourly/crypto \
    --cache-root binanceneural/forecast_cache \
    --holdout-days 3,7,14,30 \
    --fee-rate 0.001 --margin-rate 0.0625 --no-close-at-eod \
    --max-hold-hours 6 --max-positions 1 --leverage 2.0 --min-edge 0.0 \
    --no-int-qty \
    2>&1 | tail -30
  echo ""
done

echo "=== ALL DONE ==="
