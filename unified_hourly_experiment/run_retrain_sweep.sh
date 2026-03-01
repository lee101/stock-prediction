#!/bin/bash
# Retrain sweep for stocks and ETH on remote 5090
# Run both in parallel: stocks on GPU, ETH on GPU (sequential within each)
set -e
cd /nvme0n1-disk/code/stock-prediction
export PYTHONPATH=/nvme0n1-disk/code/stock-prediction

VENV=.venv312/bin/python
TRAIN="unified_hourly_experiment/train_unified_policy.py"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

# Common training params (proven from previous sweeps)
COMMON="--no-compile --no-amp --hidden-dim 512 --num-layers 6 --num-heads 8 \
  --model-arch classic --maker-fee 0.001 --margin-annual-rate 0.0625 \
  --max-leverage 2.0 --fill-buffer-pct 0.0005 --epochs 20 --batch-size 64 \
  --lr 1e-5 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 \
  --loss-type sortino --forecast-horizons 1,24"

echo "=== STOCK RETRAIN SWEEP ==="

# Config 1: Proven best (wd=0.04, rw=0.15, seq=48)
echo "[STOCK] Config 1: wd=0.04 rw=0.15 seq=48 s1337"
$VENV $TRAIN \
  --symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT \
  --crypto-symbols "" \
  --sequence-length 48 --weight-decay 0.04 --return-weight 0.15 --seed 1337 \
  --checkpoint-name retrain_stock_wd04_rw15 $COMMON

# Config 2: Higher rw (rw=0.20)
echo "[STOCK] Config 2: wd=0.04 rw=0.20 seq=48 s1337"
$VENV $TRAIN \
  --symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT \
  --crypto-symbols "" \
  --sequence-length 48 --weight-decay 0.04 --return-weight 0.20 --seed 1337 \
  --checkpoint-name retrain_stock_wd04_rw20 $COMMON

# Config 3: Lower wd (wd=0.03)
echo "[STOCK] Config 3: wd=0.03 rw=0.15 seq=48 s1337"
$VENV $TRAIN \
  --symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT \
  --crypto-symbols "" \
  --sequence-length 48 --weight-decay 0.03 --return-weight 0.15 --seed 1337 \
  --checkpoint-name retrain_stock_wd03_rw15 $COMMON

echo ""
echo "=== ETH RETRAIN SWEEP ==="

# ETH Config 1: Same proven stock params applied to ETH
echo "[ETH] Config 1: wd=0.04 rw=0.15 seq=48 s1337"
$VENV $TRAIN \
  --symbols ETHUSD \
  --stock-data-root trainingdatahourly/crypto \
  --stock-cache-root binanceneural/forecast_cache \
  --crypto-symbols "" \
  --sequence-length 48 --weight-decay 0.04 --return-weight 0.15 --seed 1337 \
  --checkpoint-name retrain_eth_wd04_rw15 $COMMON

# ETH Config 2: Higher rw for crypto volatility
echo "[ETH] Config 2: wd=0.04 rw=0.10 seq=48 s1337"
$VENV $TRAIN \
  --symbols ETHUSD \
  --stock-data-root trainingdatahourly/crypto \
  --stock-cache-root binanceneural/forecast_cache \
  --crypto-symbols "" \
  --sequence-length 48 --weight-decay 0.04 --return-weight 0.10 --seed 1337 \
  --checkpoint-name retrain_eth_wd04_rw10 $COMMON

# ETH Config 3: Longer sequence for crypto (more context)
echo "[ETH] Config 3: wd=0.04 rw=0.15 seq=72 s1337"
$VENV $TRAIN \
  --symbols ETHUSD \
  --stock-data-root trainingdatahourly/crypto \
  --stock-cache-root binanceneural/forecast_cache \
  --crypto-symbols "" \
  --sequence-length 72 --weight-decay 0.04 --return-weight 0.15 --seed 1337 \
  --checkpoint-name retrain_eth_wd04_rw15_seq72 $COMMON

# ETH Config 4: Multi-crypto (ETH + BTC for generalization)
echo "[ETH] Config 4: ETH+BTC wd=0.04 rw=0.15 seq=48 s1337"
$VENV $TRAIN \
  --symbols ETHUSD,BTCUSD \
  --stock-data-root trainingdatahourly/crypto \
  --stock-cache-root binanceneural/forecast_cache \
  --crypto-symbols "" \
  --sequence-length 48 --weight-decay 0.04 --return-weight 0.15 --seed 1337 \
  --checkpoint-name retrain_eth_btc_wd04_rw15 $COMMON

echo ""
echo "=== TRAINING COMPLETE, RUNNING EVALS ==="

# Eval all stock configs
for cfg in retrain_stock_wd04_rw15 retrain_stock_wd04_rw20 retrain_stock_wd03_rw15; do
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

# Eval all ETH configs
for cfg in retrain_eth_wd04_rw15 retrain_eth_wd04_rw10 retrain_eth_wd04_rw15_seq72 retrain_eth_btc_wd04_rw15; do
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
