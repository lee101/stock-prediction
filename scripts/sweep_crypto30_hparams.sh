#!/usr/bin/env bash
# Crypto30 hyperparameter sweep: test PPO variants to increase positive seed rate.
# Current recipe: 4/44 positive (9%) over seeds 1-50. Goal: find configs with >20% hit rate.
# Each config runs 4 seeds; eval on original crypto30 val at lag=2.
set -euo pipefail

REPO="/media/lee/pcd/code/stock"
source "$REPO/.venv313/bin/activate"
cd "$REPO"

TRAIN_BIN="$REPO/pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_BIN="$REPO/pufferlib_market/data/crypto30_daily_val.bin"
CKPT_BASE="$REPO/pufferlib_market/checkpoints/crypto30_hparam_sweep"

COMMON="--data-path $TRAIN_BIN --val-data-path $VAL_BIN \
    --max-steps 365 --trade-penalty 0.02 --disable-shorts \
    --num-envs 32 --periods-per-year 365.0 --decision-lag 2 --fee-rate 0.001 \
    --reward-scale 10.0 --reward-clip 5.0 --val-eval-windows 50"

declare -A CONFIGS

# V0: baseline (for comparison -- same as original recipe)
CONFIGS[v0_baseline]="--hidden-size 256 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 15000000"

# V1: clip_vloss + lower vf_coef (stabilize value fn)
CONFIGS[v1_clipvf]="--hidden-size 256 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 15000000 --clip-vloss --vf-coef 0.3"

# V2: relu_sq activation (proven better at small scale)
CONFIGS[v2_relusq]="--hidden-size 256 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 15000000 --arch mlp_relu_sq"

# V3: entropy annealing (start high, decay)
CONFIGS[v3_anneal_ent]="--hidden-size 256 --anneal-lr --ent-coef 0.03 --weight-decay 0.005 --total-timesteps 15000000 --anneal-ent --ent-coef-end 0.003"

# V4: shorter rollout (more frequent PPO updates)
CONFIGS[v4_short_rollout]="--hidden-size 256 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 15000000 --rollout-len 128"

# V5: longer training (30M steps)
CONFIGS[v5_30M]="--hidden-size 256 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 30000000"

# V6: shorter episodes (180 days)
CONFIGS[v6_short_ep]="--hidden-size 256 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 15000000 --max-steps 180"

# V7: resmlp arch (residual connections + LayerNorm)
CONFIGS[v7_resmlp]="--hidden-size 256 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 15000000 --arch resmlp"

# V8: higher gamma (longer horizon discount)
CONFIGS[v8_gamma995]="--hidden-size 256 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 15000000 --gamma 0.995"

# V9: kitchen sink (clip_vloss + relu_sq + anneal_ent + cosine LR)
CONFIGS[v9_combo]="--hidden-size 256 --lr-schedule cosine --ent-coef 0.02 --weight-decay 0.005 --total-timesteps 15000000 --clip-vloss --vf-coef 0.3 --arch mlp_relu_sq --anneal-ent --ent-coef-end 0.003"

# V10: wider model h512 with weaker vf
CONFIGS[v10_h512]="--hidden-size 512 --anneal-lr --ent-coef 0.01 --weight-decay 0.005 --total-timesteps 15000000 --vf-coef 0.1"

SEEDS=(1 2 3 4)

run_one() {
    local config_name=$1
    local seed=$2
    local config_args=$3
    local ckdir="${CKPT_BASE}/${config_name}/s${seed}"
    mkdir -p "$ckdir"

    if [ -f "${ckdir}/final.pt" ]; then
        echo "=== ${config_name} s${seed} already trained ==="
    else
        echo "=== Training ${config_name} s${seed} ==="
        python -u -m pufferlib_market.train \
            $COMMON $config_args \
            --seed "$seed" \
            --checkpoint-dir "$ckdir" > "${ckdir}/train.log" 2>&1 || true
    fi

    echo "=== Evaluating ${config_name} s${seed} ==="
    for pt in val_best final; do
        local ckpt="${ckdir}/${pt}.pt"
        local out_json="${ckdir}/${pt}_eval90d.json"
        [ -f "$ckpt" ] || continue
        [ -f "$out_json" ] && { echo "  ${pt} eval cached"; continue; }
        python -m pufferlib_market.evaluate_holdout \
            --checkpoint "$ckpt" \
            --data-path "$VAL_BIN" \
            --eval-hours 90 --n-windows 50 \
            --fee-rate 0.001 --fill-buffer-bps 5.0 \
            --decision-lag 2 --deterministic --no-early-stop \
            --seed 99 \
            --out "$out_json" 2>&1 | tail -3
    done
}

# Sort config names for reproducible order
sorted_configs=($(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort))

for config_name in "${sorted_configs[@]}"; do
    echo ""
    echo "========================================"
    echo "=== Config: $config_name ==="
    echo "========================================"
    for seed in "${SEEDS[@]}"; do
        run_one "$config_name" "$seed" "${CONFIGS[$config_name]}"
    done
done

echo ""
echo "========================================"
echo "=== SUMMARY ==="
echo "========================================"
for config_name in "${sorted_configs[@]}"; do
    pos=0
    total=0
    for seed in "${SEEDS[@]}"; do
        json="${CKPT_BASE}/${config_name}/s${seed}/val_best_eval90d.json"
        [ -f "$json" ] || continue
        med=$(grep median_total_return "$json" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
        sort_val=$(grep median_sortino "$json" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
        echo "${config_name} s${seed}: med=${med} sort=${sort_val}"
        ((total++))
        if (( $(echo "$med > 0" | bc -l) )); then
            ((pos++))
        fi
    done
    echo "${config_name}: ${pos}/${total} positive"
    echo "---"
done
echo "=== Done ==="
