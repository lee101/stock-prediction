#!/bin/bash
# EO-PPO finetune — warm-start from an existing v6 member (or strong standalone)
# and push it AWAY from the 12-model v7 ensemble using a small KL penalty.
#
# Rationale (2026-04-17): seed=1 (beta=0.03) and seed=2 (beta=0.005 from scratch)
# both collapsed. Training from scratch with any orthogonality overwhelms PG
# before the policy finds profitable behavior. Finetune sidesteps this — the
# warm-start policy is already profitable, so even a small beta is enough to
# create diversity without wrecking PnL.
#
# The anti-target ensemble is v7 (12 members, D_s81 dropped) — EXCLUDING the
# finetune base to avoid pulling the model away from itself in a pathological
# feedback loop.
#
# Usage:
#   nohup bash scripts/launch_eo_ppo_finetune.sh AD_s9 1 0.002 \
#     > /tmp/eo_ppo_ft_ADs9_s1_b002.log 2>&1 &
#   # args: base_name (e.g. AD_s9 or D_s64), seed, beta, total_steps, warmup_frac

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

BASE="${1:-AD_s9}"
SEED="${2:-1}"
BETA="${3:-0.002}"
TOTAL_STEPS="${4:-3000000}"
WARMUP_FRAC="${5:-0.2}"
GATE="${6:-entropy}"

# Locate the base checkpoint
BASE_CKPT=""
for cand in \
  "pufferlib_market/prod_ensemble_screened32/${BASE}.pt" \
  "pufferlib_market/checkpoints/screened32_sweep/AD/s${BASE#AD_s}/val_best.pt" \
  "pufferlib_market/checkpoints/screened32_sweep/D/s${BASE#D_s}/val_best.pt" \
  "pufferlib_market/checkpoints/screened32_sweep/C/s${BASE#C_s}/val_best.pt" \
  "pufferlib_market/checkpoints/screened32_sweep/I/s${BASE#I_s}/val_best.pt"; do
  if [ -f "$cand" ]; then BASE_CKPT="$cand"; break; fi
done
if [ -z "$BASE_CKPT" ]; then
  echo "[eo-ppo-ft] ERROR: cannot locate checkpoint for base=${BASE}"; exit 2
fi
echo "[eo-ppo-ft] base=${BASE} → ${BASE_CKPT}"

# earlyoom kills training processes under memory pressure — stop it first.
echo "ilu" | sudo -S systemctl stop earlyoom 2>/dev/null \
  && echo "[eo-ppo-ft] earlyoom stopped" \
  || echo "[eo-ppo-ft] earlyoom already stopped"

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

# v7 anti-target (12 members, D_s81 dropped). Skip whichever member matches BASE
# to avoid self-repulsion.
declare -a V7_MEMBERS=(
  "pufferlib_market/prod_ensemble_screened32/C_s7.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s16.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s42.pt"
  "pufferlib_market/prod_ensemble_screened32/AD_s4.pt"
  "pufferlib_market/prod_ensemble_screened32/I_s3.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s2.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s14.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s28.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s57.pt"
  "pufferlib_market/prod_ensemble_screened32/I_s3.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s64.pt"
  "pufferlib_market/prod_ensemble_screened32/I_s32.pt"
)
V7_CHECKPOINTS=""
for m in "${V7_MEMBERS[@]}"; do
  # skip if the basename matches the finetune base
  if [ "$(basename "$m" .pt)" = "$BASE" ]; then
    echo "[eo-ppo-ft] excluding ${m} from anti-target (matches base)"
    continue
  fi
  if [ -z "$V7_CHECKPOINTS" ]; then V7_CHECKPOINTS="$m"; else V7_CHECKPOINTS="$V7_CHECKPOINTS,$m"; fi
done

TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_augmented_val.bin"

TAG="ft_${BASE}_s${SEED}_b${BETA}"
CKPT_DIR="pufferlib_market/checkpoints/eo_ppo_pilot/${TAG}"
mkdir -p "$CKPT_DIR"

echo "[eo-ppo-ft] seed=${SEED} beta=${BETA} steps=${TOTAL_STEPS} warmup=${WARMUP_FRAC} gate=${GATE}"
echo "[eo-ppo-ft] ckpt_dir=${CKPT_DIR}"

python -u -m pufferlib_market.train \
    --data-path "$TRAIN" \
    --val-data-path "$VAL" \
    --resume-from "$BASE_CKPT" \
    --total-timesteps "$TOTAL_STEPS" \
    --max-steps 252 \
    --trade-penalty 0.05 \
    --hidden-size 1024 \
    --anneal-lr \
    --disable-shorts \
    --val-eval-windows 50 \
    --early-stop-val-neg-threshold 25 \
    --early-stop-val-neg-patience 2 \
    --optimizer muon \
    --num-envs 128 \
    --seed "$SEED" \
    --checkpoint-dir "$CKPT_DIR" \
    --ensemble-kl-beta "$BETA" \
    --ensemble-kl-checkpoints "$V7_CHECKPOINTS" \
    --ensemble-kl-gate "$GATE" \
    --ensemble-kl-warmup-frac "$WARMUP_FRAC" \
    --ensemble-kl-start-frac 0.0 \
    2>&1 | tee "$CKPT_DIR/train.log"

echo "[eo-ppo-ft] done. Gate eval:"
echo "  python scripts/eval_multihorizon_candidate.py \\"
echo "    --candidate-checkpoint $CKPT_DIR/val_best.pt \\"
echo "    --data-path pufferlib_market/data/screened32_single_offset_val_full.bin \\"
echo "    --horizons-days 30,100 --slippage-bps 5 --fill-buffer-bps 5 \\"
echo "    --decision-lag 2 --recent-within-days 140 --n-windows 24 \\"
echo "    --out reports/eo_ppo_${TAG}_14th.json"
