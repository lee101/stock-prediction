#!/bin/bash
# Train Chronos2 LoRA adapters for the top 5 priority symbols identified by
# the MAE dashboard (chronos2_mae_dashboard.csv).
#
# Symbols:  AVAXUSD (crypto), NET, AMD, COIN, EXPE (stocks)
# Sweep:    preaug=differencing, lora_r={16,32}, lr={1e-5,5e-5},
#           context_length={256,512}, prediction_length=1, num_steps=1000
# Total:    8 configs per symbol, 40 runs (~1-2 min each)
#
# NOTE: COIN has only ~479 hourly rows -- too few for ctx>=256 with default
#       val/test=168h each.  Those configs will fail gracefully and the error
#       is printed but does not abort the sweep.
#
# Results:  hyperparams/crypto_lora_sweep/priority_batch/
# Models:   chronos2_finetuned/priority_batch/

set -euo pipefail

cd /nvme0n1-disk/code/stock-prediction
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate

RESULTS_DIR="hyperparams/crypto_lora_sweep/priority_batch"
OUTPUT_ROOT="chronos2_finetuned/priority_batch"
RUN_PREFIX="priority"
PREAUG="differencing"
PREDICTION_LENGTH=1
NUM_STEPS=1000
BATCH_SIZE=32

mkdir -p "$RESULTS_DIR" "$OUTPUT_ROOT"

# Map each symbol to its data root directory.
declare -A DATA_ROOTS=(
    [AVAXUSD]="trainingdatahourly/crypto"
    [NET]="trainingdatahourly/stocks"
    [AMD]="trainingdatahourly/stocks"
    [COIN]="trainingdatahourly/stocks"
    [EXPE]="trainingdatahourly/stocks"
)

SYMBOLS=(AVAXUSD NET AMD COIN EXPE)
LORA_RS=(16 32)
LRS=(1e-5 5e-5)
CTXS=(256 512)

TOTAL=$(( ${#SYMBOLS[@]} * ${#LORA_RS[@]} * ${#LRS[@]} * ${#CTXS[@]} ))
IDX=0
PASS=0
FAIL=0

for SYM in "${SYMBOLS[@]}"; do
    DATA_ROOT="${DATA_ROOTS[$SYM]}"
    for R in "${LORA_RS[@]}"; do
        for LR in "${LRS[@]}"; do
            for CTX in "${CTXS[@]}"; do
                IDX=$((IDX + 1))
                echo ""
                echo "=== [$IDX/$TOTAL] $SYM  r=$R  lr=$LR  ctx=$CTX  preaug=$PREAUG ==="
                if python scripts/train_crypto_lora_sweep.py \
                    --symbol "$SYM" \
                    --data-root "$DATA_ROOT" \
                    --output-root "$OUTPUT_ROOT" \
                    --results-dir "$RESULTS_DIR" \
                    --context-length "$CTX" \
                    --prediction-length "$PREDICTION_LENGTH" \
                    --batch-size "$BATCH_SIZE" \
                    --learning-rate "$LR" \
                    --num-steps "$NUM_STEPS" \
                    --lora-r "$R" \
                    --preaug "$PREAUG" \
                    --run-prefix "$RUN_PREFIX"; then
                    PASS=$((PASS + 1))
                else
                    FAIL=$((FAIL + 1))
                    echo "  !! FAILED: $SYM r=$R lr=$LR ctx=$CTX"
                fi
            done
        done
    done
done

echo ""
echo "=== Sweep complete: $PASS passed, $FAIL failed out of $TOTAL ==="
echo "=== Results in $RESULTS_DIR ==="
