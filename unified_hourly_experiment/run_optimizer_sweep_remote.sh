#!/bin/bash
set -e
cd /home/lee/code/stock
export PYTHONUNBUFFERED=1

REMOTE="sshpass -p 'ka3iMI4OSNvgFcREuDaQyLguFxuP' ssh -o StrictHostKeyChecking=no administrator@93.127.141.100"
REMOTE_DIR="/nvme0n1-disk/code/stock-prediction"

echo "=== Syncing code to remote ==="
sshpass -p 'ka3iMI4OSNvgFcREuDaQyLguFxuP' rsync -avz --exclude='.venv*' --exclude='__pycache__' --exclude='*.pt' --exclude='checkpoints' --exclude='modded-nanogpt' \
    -e "ssh -o StrictHostKeyChecking=no" \
    /home/lee/code/stock/{binanceneural,unified_hourly_experiment,differentiable_loss_utils.py,traininglib,src} \
    administrator@93.127.141.100:$REMOTE_DIR/

# Run remote sweep - complementary configs to local
$REMOTE "cd $REMOTE_DIR && bash -c '
export PYTHONUNBUFFERED=1
PYTHON=\".venv312/bin/python -u\"
TRAIN=\"unified_hourly_experiment/train_unified_policy.py\"
SWEEP=\"unified_hourly_experiment/sweep_epoch_portfolio.py\"

COMMON=\"--symbols NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT --crypto-symbols \\\"\\\" --epochs 20 --batch-size 64 --sequence-length 48 --hidden-dim 512 --num-layers 6 --num-heads 8 --seed 1337 --warmup-steps 100 --dropout 0.1 --grad-clip 1.0 --fill-temperature 5e-4 --logits-softcap 12.0 --decision-lag-bars 1 --forecast-horizons 1 --no-compile --maker-fee 0.001 --max-leverage 2.0 --margin-annual-rate 0.0625 --fill-buffer-pct 0.0005 --validation-use-binary-fills --return-weight 0.15\"

EVAL_BASE=\"--symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT --blacklist EBAY,NET --fee-rate 0.001 --margin-rate 0.0625 --no-close-at-eod --max-positions 7 --max-hold-hours 6 --min-edge 0.0\"

# Remote runs different seed + Muon lr combos
echo \"=== R1. Muon lr=0.02 seed=42 ===\"
\$PYTHON \$TRAIN \$COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --seed 42 --checkpoint-name opt_muon_s42
\$PYTHON \$SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/opt_muon_s42 \$EVAL_BASE --holdout-days 30 2>&1 | tail -5

echo \"=== R2. Muon lr=0.02 + cautious WD seed=42 ===\"
\$PYTHON \$TRAIN \$COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --cautious-wd --seed 42 --checkpoint-name opt_muon_cautious_s42
\$PYTHON \$SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/opt_muon_cautious_s42 \$EVAL_BASE --holdout-days 30 2>&1 | tail -5

echo \"=== R3. Muon lr=0.01 seed=1337 ===\"
\$PYTHON \$TRAIN \$COMMON --optimizer muon --muon-lr 0.01 --lr 1e-4 --weight-decay 0.03 --checkpoint-name opt_muon_001_r
\$PYTHON \$SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/opt_muon_001_r \$EVAL_BASE --holdout-days 30 2>&1 | tail -5

echo \"=== R4. Muon + all features seed=42 ===\"
\$PYTHON \$TRAIN \$COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --cautious-wd --cooldown-fraction 0.2 --embed-lr-mult 5.0 --seed 42 --checkpoint-name opt_muon_all_s42
\$PYTHON \$SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/opt_muon_all_s42 \$EVAL_BASE --holdout-days 30 2>&1 | tail -5

echo \"=== R5. AdamW lr=1e-5 wd=0.05 (deployed-like higher wd) ===\"
\$PYTHON \$TRAIN \$COMMON --lr 1e-5 --weight-decay 0.05 --checkpoint-name opt_adamw_wd05
\$PYTHON \$SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/opt_adamw_wd05 \$EVAL_BASE --holdout-days 30 2>&1 | tail -5

echo \"=== R6. Muon lr=0.02 + LR warmdown ===\"
\$PYTHON \$TRAIN \$COMMON --optimizer muon --muon-lr 0.02 --lr 1e-4 --weight-decay 0.03 --lr-schedule linear_warmdown --checkpoint-name opt_muon_warmdown
\$PYTHON \$SWEEP --checkpoint-dir unified_hourly_experiment/checkpoints/opt_muon_warmdown \$EVAL_BASE --holdout-days 30 2>&1 | tail -5

echo \"=== Remote sweep complete ===\"
'"
