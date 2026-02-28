#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

PYTHON=".venv312/bin/python -u"
TRAIN="unified_hourly_experiment/train_unified_policy.py"
SWEEP="unified_hourly_experiment/sweep_epoch_portfolio.py"

# Base config matching deployed (realistic_rw015 ep7)
COMMON="--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --crypto-symbols '' --epochs 20 --batch-size 64 --sequence-length 48 --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 1337 --warmup-steps 100 --dropout 0.1 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 --decision-lag-bars 1 --forecast-horizons 1 --no-compile --maker-fee 0.001 --max-leverage 2.0 --margin-annual-rate 0.0625 --fill-buffer-pct 0.0005 --validation-use-binary-fills --return-weight 0.15"

EVAL_BASE="--symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT --blacklist EBAY,NET --fee-rate 0.001 --margin-rate 0.0625 --no-close-at-eod --max-positions 7 --max-hold-hours 6 --min-edge 0.0"

eval_run() {
    local name=$1
    echo "--- Eval: $name ---"
    for days in 30 60; do
        echo "  holdout=${days}d:"
        $PYTHON $SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/$name $EVAL_BASE --holdout-days $days 2>&1 | tail -5
    done
}

# 1. Baseline: AdamW + BF16 (current deployed config)
echo "=== 1. Baseline: AdamW lr=1e-5 wd=0.03 BF16 ==="
$PYTHON $TRAIN $COMMON --lr 1e-5 --weight-decay 0.03 --checkpoint-name opt_baseline
eval_run opt_baseline

# 2. AdamW no AMP (to measure BF16 impact)
echo "=== 2. AdamW no AMP ==="
$PYTHON $TRAIN $COMMON --lr 1e-5 --weight-decay 0.03 --no-amp --checkpoint-name opt_noamp
eval_run opt_noamp

# 3. Muon + BF16
echo "=== 3. Muon lr=0.02 ==="
$PYTHON $TRAIN $COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --checkpoint-name opt_muon_002
eval_run opt_muon_002

# 4. Muon lr sweep
for MLR in 0.005 0.01 0.04; do
    echo "=== 4. Muon lr=$MLR ==="
    $PYTHON $TRAIN $COMMON --optimizer muon --muon-lr $MLR --lr 1e-4 --weight-decay 0.03 --checkpoint-name opt_muon_${MLR}
    eval_run opt_muon_${MLR}
done

# 5. Muon + cautious WD
echo "=== 5. Muon + cautious WD ==="
$PYTHON $TRAIN $COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --cautious-wd --checkpoint-name opt_muon_cautious
eval_run opt_muon_cautious

# 6. Muon + cooldown
echo "=== 6. Muon + cooldown 20% ==="
$PYTHON $TRAIN $COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --cooldown-fraction 0.2 --checkpoint-name opt_muon_cooldown
eval_run opt_muon_cooldown

# 7. Muon + per-param LR (higher embed LR)
echo "=== 7. Muon + embed-lr-mult=5 ==="
$PYTHON $TRAIN $COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --embed-lr-mult 5.0 --checkpoint-name opt_muon_embedlr5
eval_run opt_muon_embedlr5

# 8. Muon + all features combined
echo "=== 8. Muon + all ==="
$PYTHON $TRAIN $COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --cautious-wd --cooldown-fraction 0.2 --embed-lr-mult 5.0 --checkpoint-name opt_muon_all
eval_run opt_muon_all

# 9. Muon momentum sweep
for MOM in 0.90 0.99; do
    echo "=== 9. Muon momentum=$MOM ==="
    $PYTHON $TRAIN $COMMON --optimizer muon --muon-lr 0.02 --muon-momentum $MOM --lr 1e-4 --weight-decay 0.03 --checkpoint-name opt_muon_mom${MOM}
    eval_run opt_muon_mom${MOM}
done

# 10. AdamW with higher LR (to see if BF16 unlocks higher LR)
echo "=== 10. AdamW lr=3e-5 ==="
$PYTHON $TRAIN $COMMON --lr 3e-5 --weight-decay 0.03 --checkpoint-name opt_adamw_3e5
eval_run opt_adamw_3e5

echo "=== All sweeps complete ==="
