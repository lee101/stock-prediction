#!/bin/bash
set -e
cd /home/lee/code/stock
source .venv/bin/activate

SYMBOLS="NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT"
COMMON="--symbols $SYMBOLS --epochs 50 --batch-size 64 --sequence-length 32 --hidden-dim 512 --num-layers 6 --num-heads 8 --return-weight 0.15 --weight-decay 0.04 --lr 1e-5 --seed 1337 --warmup-steps 100 --decision-lag-bars 1 --fill-temperature 5e-4 --maker-fee 0.001"

echo "=== A1: Muon optimizer ==="
python unified_hourly_experiment/train_unified_policy.py $COMMON \
  --model-arch classic --optimizer muon_mix --muon-lr 0.02 --muon-momentum 0.95 \
  --checkpoint-name sweep_muon_rw015 2>&1

echo "=== A2: Nano arch (RoPE, MQA, dilated, memory tokens) ==="
python unified_hourly_experiment/train_unified_policy.py $COMMON \
  --model-arch nano --optimizer adamw \
  --dilated-strides "1,4,8" --num-memory-tokens 4 --use-value-embedding \
  --num-kv-heads 2 \
  --checkpoint-name sweep_nano_full 2>&1

echo "=== A3: Calmar loss ==="
python unified_hourly_experiment/train_unified_policy.py $COMMON \
  --model-arch classic --optimizer adamw \
  --loss-type calmar --dd-penalty 1.0 \
  --checkpoint-name sweep_calmar_rw015 2>&1

echo "=== A4: Multi-lag training ==="
python unified_hourly_experiment/train_unified_policy.py $COMMON \
  --model-arch classic --optimizer adamw \
  --decision-lag-range "0,1,2" \
  --checkpoint-name sweep_multilag_rw015 2>&1

echo "=== All Group A training complete ==="
