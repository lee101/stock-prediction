#!/bin/bash
# EO-PPO pilot — train a D-recipe specialist with ensemble-orthogonality
# loss anti-targetting the 13-model v6 prod ensemble.
#
# See docs/ensemble_orthogonal_ppo.md for algorithm rationale.
#
# Usage:
#   nohup bash scripts/launch_eo_ppo_pilot.sh 1 0.03 > /tmp/eo_ppo_s1_b003.log 2>&1 &
#   # args: seed (default 1), beta (default 0.03)

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

SEED="${1:-1}"
BETA="${2:-0.03}"
WARMUP_FRAC="${3:-0.3}"
GATE="${4:-entropy}"
START_FRAC="${5:-0.0}"
VAL_WINDOWS="${6:-50}"
VAL_DATA_OVERRIDE="${7:-}"

# earlyoom kills training processes under memory pressure — stop it first.
echo "ilu" | sudo -S systemctl stop earlyoom 2>/dev/null \
  && echo "[eo-ppo] earlyoom stopped" \
  || echo "[eo-ppo] earlyoom already stopped"

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

# v7 prod ensemble (12 members). Matches src/daily_stock_defaults as of 2026-04-17.
# C_s7 anchor + 11 extras (D_s81 dropped on LOO signal; I_s32 swapped in for D_s27).
# I_s3 duplicated = 2x weight (bear-resistant). Anti-target for EO-PPO = same list.
PROD_CHECKPOINTS="pufferlib_market/prod_ensemble_screened32/C_s7.pt,\
pufferlib_market/prod_ensemble_screened32/D_s16.pt,\
pufferlib_market/prod_ensemble_screened32/D_s42.pt,\
pufferlib_market/prod_ensemble_screened32/AD_s4.pt,\
pufferlib_market/prod_ensemble_screened32/I_s3.pt,\
pufferlib_market/prod_ensemble_screened32/D_s2.pt,\
pufferlib_market/prod_ensemble_screened32/D_s14.pt,\
pufferlib_market/prod_ensemble_screened32/D_s28.pt,\
pufferlib_market/prod_ensemble_screened32/D_s57.pt,\
pufferlib_market/prod_ensemble_screened32/I_s3.pt,\
pufferlib_market/prod_ensemble_screened32/D_s64.pt,\
pufferlib_market/prod_ensemble_screened32/I_s32.pt"

TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="${VAL_DATA_OVERRIDE:-pufferlib_market/data/screened32_augmented_val.bin}"

TAG="b${BETA}_s${SEED}"
if [ "${START_FRAC}" != "0.0" ]; then
  TAG="${TAG}_sf${START_FRAC}"
fi
if [ "${VAL_WINDOWS}" != "50" ]; then
  TAG="${TAG}_vw${VAL_WINDOWS}"
fi
CKPT_DIR="pufferlib_market/checkpoints/eo_ppo_pilot/${TAG}"
mkdir -p "$CKPT_DIR"

echo "[eo-ppo] seed=${SEED} beta=${BETA} warmup=${WARMUP_FRAC} start=${START_FRAC} gate=${GATE}"
echo "[eo-ppo] ckpt_dir=${CKPT_DIR}"

python -u -m pufferlib_market.train \
    --data-path "$TRAIN" \
    --val-data-path "$VAL" \
    --total-timesteps 15000000 \
    --max-steps 252 \
    --trade-penalty 0.05 \
    --hidden-size 1024 \
    --anneal-lr \
    --disable-shorts \
    --val-eval-windows "$VAL_WINDOWS" \
    --early-stop-val-neg-threshold 25 \
    --early-stop-val-neg-patience 2 \
    --optimizer muon \
    --num-envs 128 \
    --seed "$SEED" \
    --checkpoint-dir "$CKPT_DIR" \
    --ensemble-kl-beta "$BETA" \
    --ensemble-kl-checkpoints "$PROD_CHECKPOINTS" \
    --ensemble-kl-gate "$GATE" \
    --ensemble-kl-warmup-frac "$WARMUP_FRAC" \
    --ensemble-kl-start-frac "$START_FRAC" \
    2>&1 | tee "$CKPT_DIR/train.log"

echo "[eo-ppo] done. Gate eval next:"
echo "  python scripts/eval_multihorizon_candidate.py \\"
echo "    --candidate-checkpoint $CKPT_DIR/val_best.pt \\"
echo "    --out reports/eo_ppo_b${BETA}_s${SEED}_14th.json"
