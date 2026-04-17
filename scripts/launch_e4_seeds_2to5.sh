#!/bin/bash
# Launch E4 (D variant, lev2x_ds03) seeds 2-5 in parallel with isolated
# TMPDIRs. s1 already REJECT-SOFT (+0.00 neg, -0.56% med); looking for a
# seed that also lifts median.
set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

TRAIN=pufferlib_market/data/screened32_augmented_train.bin
VAL=pufferlib_market/data/screened32_augmented_val.bin
CKPT_ROOT=pufferlib_market/checkpoints/screened32_leverage_sweep/D/lev2x_ds03

PIDS=()
for SEED in 2 3 4 5; do
  DIR="${CKPT_ROOT}/s${SEED}"
  TMP="/nvme0n1-disk/code/stock-prediction/.tmp_train/e4_lev2x_s${SEED}"
  mkdir -p "$DIR" "$TMP"
  echo "[launch] E4 D lev2x_ds03 seed=${SEED} dir=${DIR} tmp=${TMP}"
  TMPDIR="$TMP" nohup python -u -m pufferlib_market.train \
      --data-path "$TRAIN" --val-data-path "$VAL" \
      --total-timesteps 15000000 --max-steps 252 \
      --trade-penalty 0.05 --hidden-size 1024 \
      --anneal-lr --disable-shorts --num-envs 128 \
      --val-eval-windows 30 \
      --max-leverage 2.0 --downside-penalty 0.3 \
      --optimizer muon \
      --seed "$SEED" --checkpoint-dir "$DIR" \
      > "$DIR/train.log" 2>&1 &
  PID=$!
  PIDS+=("$PID")
  echo "[launched] seed=${SEED} pid=${PID}"
  sleep 3
done
echo "[all launched] pids=${PIDS[*]}"
echo "${PIDS[*]}" > /nvme0n1-disk/code/stock-prediction/.tmp_train/e4_seeds_2to5.pids
