#!/bin/bash
# Train architecture-diverse models for meta-selector
# Run AFTER train_meta_diverse.sh completes
set +e

DATA=pufferlib_market/data/stocks12_fresh_train.bin
VAL=pufferlib_market/data/stocks12_fresh_val.bin
BASE_DIR=pufferlib_market/checkpoints/meta_diverse
STEPS=10000000

source .venv/bin/activate

declare -a CONFIGS=(
    "a1_resmlp_s1_h1024     --seed 21 --arch resmlp --hidden-size 1024 --weight-decay 0.03 --trade-penalty 0.05"
    "a2_resmlp_s2_h512      --seed 22 --arch resmlp --hidden-size 512  --weight-decay 0.0  --trade-penalty 0.02"
    "a3_transformer_s1_h256 --seed 23 --arch transformer --hidden-size 256 --weight-decay 0.03 --trade-penalty 0.05"
    "a4_transformer_s2_h512 --seed 24 --arch transformer --hidden-size 512 --weight-decay 0.0  --trade-penalty 0.02"
    "a5_gru_s1_h512         --seed 25 --arch gru --hidden-size 512 --weight-decay 0.03 --trade-penalty 0.05"
    "a6_gru_s2_h256         --seed 26 --arch gru --hidden-size 256 --weight-decay 0.0  --trade-penalty 0.02"
    "a7_depth_s1_h512       --seed 27 --arch depth_recurrence --hidden-size 512 --weight-decay 0.03 --trade-penalty 0.05"
    "a8_depth_s2_h256       --seed 28 --arch depth_recurrence --hidden-size 256 --weight-decay 0.0  --trade-penalty 0.02"
)

for cfg in "${CONFIGS[@]}"; do
    NAME=$(echo "$cfg" | awk '{print $1}')
    ARGS=$(echo "$cfg" | cut -d' ' -f2-)
    DIR="${BASE_DIR}/${NAME}"

    if [ -f "${DIR}/final.pt" ]; then
        echo "SKIP $NAME (final.pt exists)"
        continue
    fi

    echo "=== Training $NAME ==="
    mkdir -p "$DIR"
    python -u -m pufferlib_market.train \
        --data-path "$DATA" \
        --val-data-path "$VAL" \
        --total-timesteps "$STEPS" \
        --max-steps 252 \
        --anneal-lr \
        --disable-shorts \
        --num-envs 128 \
        --checkpoint-dir "$DIR" \
        $ARGS \
        2>&1 | tee "${DIR}/train.log"

    echo "=== Done $NAME ==="
done

echo "All arch training complete."
