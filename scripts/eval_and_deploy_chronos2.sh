#!/usr/bin/env bash
# ============================================================================
# Evaluate a finetuned Chronos2 checkpoint and optionally deploy it.
#
# Usage:
#   bash scripts/eval_and_deploy_chronos2.sh \
#       --ckpt chronos2_finetuned/stocks_all_v2/finetuned-ckpt \
#       [--cache-path .cache/chronos2_train_data_full.npz] \
#       [--baseline-mae-pct 2.45] \
#       [--deploy]
#
# Steps:
#   1. Quick MAE eval on finetuned checkpoint (uses full cache or defaults)
#   2. Compare to baseline (default amazon/chronos-2, MAE%=2.45%)
#   3. If improved (or --force-deploy): run benchmark on key symbols
#   4. Update hyperparams/chronos2/*.json to use finetuned model
#   5. Run linear calibration for buy/sell thresholds
# ============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# --- Defaults ---
CKPT=""
CACHE_PATH=".cache/chronos2_train_data_full.npz"
BASELINE_MAE_PCT=2.45
DEPLOY=false
FORCE_DEPLOY=false
MAX_SHIFT_BPS=8
ALLOW_SHORT=false

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)              CKPT=$2;               shift 2 ;;
        --cache-path)        CACHE_PATH=$2;         shift 2 ;;
        --baseline-mae-pct)  BASELINE_MAE_PCT=$2;   shift 2 ;;
        --deploy)            DEPLOY=true;           shift   ;;
        --force-deploy)      FORCE_DEPLOY=true; DEPLOY=true; shift ;;
        --max-shift-bps)     MAX_SHIFT_BPS=$2;      shift 2 ;;
        --allow-short)       ALLOW_SHORT=true;      shift   ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "ERROR: --ckpt is required (e.g. chronos2_finetuned/stocks_all_v1/finetuned-ckpt)"
    exit 1
fi

if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: checkpoint dir not found: $CKPT"
    exit 1
fi

source .venv/bin/activate

CKPT_BASE="$(basename "$(dirname "$CKPT")")"

echo "=== Chronos2 eval & deploy: $CKPT ==="
echo "  baseline MAE% threshold: $BASELINE_MAE_PCT"
echo "  cache: $CACHE_PATH"
echo "======================================"

# --- Step 1: Quick MAE eval ---
echo ""
echo "[1/4] Quick MAE eval on finetuned model..."
EVAL_OUT=$(python chronos2_full_finetune.py \
    --eval-mae-only \
    --model-id "$CKPT" \
    --cache-path "$CACHE_PATH" \
    2>&1)
echo "$EVAL_OUT"

# Extract MAE% from output (format: "Baseline val MAE: X.XXXX  MAE%: Y.YY%  (n=N)")
FT_MAE_PCT=$(echo "$EVAL_OUT" | grep -oP "MAE%: \K[0-9]+\.[0-9]+" | head -1)
if [[ -z "$FT_MAE_PCT" ]]; then
    echo "ERROR: Could not extract MAE% from eval output"
    exit 1
fi
echo ""
echo "Finetuned MAE%: $FT_MAE_PCT%"
echo "Baseline MAE%:  $BASELINE_MAE_PCT%"

# Compare (using awk for float comparison)
IMPROVED=$(awk -v ft="$FT_MAE_PCT" -v base="$BASELINE_MAE_PCT" 'BEGIN{print (ft < base) ? "yes" : "no"}')
echo "Improved: $IMPROVED"

if [[ "$IMPROVED" != "yes" && "$FORCE_DEPLOY" != "true" ]]; then
    echo ""
    echo "SKIP: finetuned MAE% ($FT_MAE_PCT%) >= baseline ($BASELINE_MAE_PCT%). Not deploying."
    echo "      Use --force-deploy to override."
    exit 0
fi

echo ""
echo "Model improved! MAE% $BASELINE_MAE_PCT% → $FT_MAE_PCT%"

if [[ "$DEPLOY" != "true" ]]; then
    echo "Dry run (pass --deploy to actually deploy)"
    exit 0
fi

# --- Step 2: Run calibration ---
echo ""
echo "[2/4] Running linear threshold calibration..."
CAL_ARGS=(
    --model-id     "$CKPT"
    --cal-data-dir trainingdata
    --output-path  "${CKPT}/calibration.json"
    --max-shift-bps "$MAX_SHIFT_BPS"
    --grid-steps   25
)
if $ALLOW_SHORT; then
    CAL_ARGS+=(--allow-short)
fi
python chronos2_linear_calibration.py "${CAL_ARGS[@]}"

# --- Step 3: Benchmark on key symbols ---
echo ""
echo "[3/4] Running benchmark on key symbols..."
BENCH_SYMBOLS="AAPL,GOOG,TSLA,MSFT,AMZN,SPY,QQQ"
python benchmark_chronos2.py \
    --symbols "$BENCH_SYMBOLS" \
    --model-id "$CKPT" \
    --update-hyperparams \
    2>&1 | tail -30

# --- Step 4: Update alpacaprod.md ---
echo ""
echo "[4/4] Deployment summary:"
echo "  Checkpoint: $CKPT"
echo "  MAE%: $BASELINE_MAE_PCT% → $FT_MAE_PCT% (delta: $(awk -v ft=$FT_MAE_PCT -v b=$BASELINE_MAE_PCT 'BEGIN{printf "%+.2f", ft-b}')%)"
echo "  Calibration: ${CKPT}/calibration.json"
echo ""
echo "IMPORTANT: Update alpacaprod.md with these results and restart the Chronos2 trading service."
echo ""
echo "To use the finetuned model in production, set:"
echo "  export CHRONOS2_MODEL_ID_OVERRIDE='$CKPT'"
echo "  Or update hyperparams/chronos2/*.json model_id field"
