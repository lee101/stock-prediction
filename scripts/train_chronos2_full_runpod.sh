#!/usr/bin/env bash
# ============================================================================
# Train full Chronos2 on all stock/crypto data — RunPod launcher
#
# Usage (from project root on RunPod or locally):
#   bash scripts/train_chronos2_full_runpod.sh [options]
#
# Options:
#   --steps N         Training steps (default: 100000)
#   --batch N         Batch size (default: 512 for A100, 256 for smaller)
#   --context N       Context length (default: 512)
#   --lr FLOAT        Peak learning rate (default: 5e-5)
#   --lora            Use LoRA fine-tuning instead of full
#   --muon            Use Muon optimizer (default: enabled)
#   --no-muon         Disable Muon, use AdamW instead
#   --grad-accum N    Gradient accumulation steps (default: 1)
#   --r2-prefix STR   R2 key prefix for checkpoint upload
#   --output-dir STR  Local output directory
#   --cache STR       Path to pre-built .npz cache
#   --rebuild-cache   Force rebuild data cache even if it exists
#   --tag STR         Version tag for naming (default: v3)
#
# v4 example (ctx=1024, larger effective batch):
#   bash scripts/train_chronos2_full_runpod.sh \
#       --tag v4 --context 1024 --batch 128 --grad-accum 2 --steps 200000
#
# Environment vars (required for R2 upload):
#   R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY, R2_SECRET_KEY
#   (or CLOUDFLARE_R2_* variants set on this machine)
#
# The script:
#   1. Installs deps if on fresh RunPod pod
#   2. Activates virtual env
#   3. Builds/updates data cache if stale
#   4. Runs chronos2_full_finetune.py
#   5. Uploads checkpoint + calibration to R2
#   6. Runs post-training calibration (buy/sell thresholds +/-8bps)
#   7. Prints final MAE comparison vs baseline
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

# --- Defaults ---
NUM_STEPS=100000
BATCH_SIZE=512
CONTEXT=512
LR=5e-5
FINETUNE_MODE=full
LORA_R=32
GRAD_ACCUM=1
AMP_LOG_STD=0.45
NOISE_FRAC=0.003
DROPOUT_RATE=0.03
CHANNEL_DROPOUT=0.0
TIME_WARP=0.0
SEED=42
USE_MUON=true          # Muon on by default for RunPod runs
R2_PREFIX=""           # set auto from tag below
CACHE_PATH=".cache/chronos2_train_data_full.npz"
OUTPUT_DIR=""          # auto-named from tag
TAG="v3"
REBUILD_CACHE=false
WANDB_PROJECT=""
WANDB_RUN=""

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)         NUM_STEPS=$2;       shift 2 ;;
        --batch)         BATCH_SIZE=$2;      shift 2 ;;
        --context)       CONTEXT=$2;         shift 2 ;;
        --lr)            LR=$2;              shift 2 ;;
        --lora)          FINETUNE_MODE=lora; shift   ;;
        --muon)          USE_MUON=true;      shift   ;;
        --no-muon)       USE_MUON=false;     shift   ;;
        --grad-accum)    GRAD_ACCUM=$2;      shift 2 ;;
        --amp-log-std)        AMP_LOG_STD=$2;       shift 2 ;;
        --noise-frac)         NOISE_FRAC=$2;        shift 2 ;;
        --dropout-rate)       DROPOUT_RATE=$2;      shift 2 ;;
        --channel-dropout)    CHANNEL_DROPOUT=$2;   shift 2 ;;
        --time-warp)          TIME_WARP=$2;         shift 2 ;;
        --seed)               SEED=$2;              shift 2 ;;
        --r2-prefix)     R2_PREFIX=$2;       shift 2 ;;
        --output-dir)    OUTPUT_DIR=$2;      shift 2 ;;
        --cache)         CACHE_PATH=$2;      shift 2 ;;
        --rebuild-cache) REBUILD_CACHE=true; shift   ;;
        --tag)           TAG=$2;             shift 2 ;;
        --wandb-project) WANDB_PROJECT=$2;   shift 2 ;;
        --wandb-run)     WANDB_RUN=$2;       shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --- Auto-set names from tag ---
[[ -z "$OUTPUT_DIR" ]]  && OUTPUT_DIR="chronos2_finetuned/stocks_all_${TAG}"
[[ -z "$R2_PREFIX" ]]   && R2_PREFIX="chronos2/finetune/stocks_all_${TAG}"

# --- Map CLOUDFLARE_R2_* -> R2_* if not already set ---
if [[ -z "${R2_ACCESS_KEY:-}" && -n "${CLOUDFLARE_R2_ACCESS_KEY_ID:-}" ]]; then
    export R2_ACCESS_KEY="$CLOUDFLARE_R2_ACCESS_KEY_ID"
fi
if [[ -z "${R2_SECRET_KEY:-}" && -n "${CLOUDFLARE_R2_SECRET_ACCESS_KEY:-}" ]]; then
    export R2_SECRET_KEY="$CLOUDFLARE_R2_SECRET_ACCESS_KEY"
fi
if [[ -z "${R2_BUCKET:-}" ]]; then
    export R2_BUCKET="models"
fi

# --- Install deps on fresh pod (if setup.py/pyproject.toml not yet installed) ---
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "=== RunPod pod detected: $RUNPOD_POD_ID ==="
    if ! python -c "import chronos" 2>/dev/null; then
        echo "Installing dependencies..."
        pip install uv
        uv pip install -e chronos-forecasting/ --quiet
        uv pip install peft boto3 --quiet
    fi
fi

# --- Activate venv ---
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f ".venv312/bin/activate" ]]; then
    source .venv312/bin/activate
fi

echo "=== Chronos2 full fine-tune (${TAG}) ==="
echo "  steps=$NUM_STEPS batch=$BATCH_SIZE ctx=$CONTEXT lr=$LR"
echo "  mode=$FINETUNE_MODE muon=$USE_MUON"
echo "  output=$OUTPUT_DIR"
echo "  cache=$CACHE_PATH"
echo "  r2=$R2_PREFIX (bucket=${R2_BUCKET:-unset})"
echo "=========================================="

# --- Cache: download from R2, or rebuild if needed ---
R2_CACHE_KEY="chronos2/data/chronos2_train_data_full_v2.npz"
if [[ "$REBUILD_CACHE" == "true" ]]; then
    [[ -f "$CACHE_PATH" ]] && rm "$CACHE_PATH"
fi
if [[ ! -f "$CACHE_PATH" ]]; then
    # Try downloading from R2 first (fast: ~330MB vs ~5min rebuild)
    if [[ -n "${R2_ENDPOINT:-}" && -n "${R2_ACCESS_KEY:-}" ]]; then
        echo "Downloading data cache from R2 ($R2_CACHE_KEY)..."
        mkdir -p "$(dirname "$CACHE_PATH")"
        python - <<PYEOF
import os, sys
sys.path.insert(0, '.')
from src.r2_client import R2Client
from pathlib import Path
try:
    client = R2Client()
    client.download_file('$R2_CACHE_KEY', '$CACHE_PATH')
    print('Downloaded cache from R2: $CACHE_PATH')
except Exception as e:
    print(f'R2 download failed: {e}')
    sys.exit(1)
PYEOF
        CACHE_DL_STATUS=$?
    else
        CACHE_DL_STATUS=1
    fi

    if [[ "$CACHE_DL_STATUS" != "0" || ! -f "$CACHE_PATH" ]]; then
        echo "Building data cache from scratch: $CACHE_PATH ..."
        python -c "
from chronos2_stock_augmentation import prepare_all_training_series, AugConfig
from pathlib import Path
cfg = AugConfig(add_return_variants=True, sliding_daily_offsets=list(range(7)))
result = prepare_all_training_series(
    daily_data_dir=Path('trainingdata'),
    hourly_data_dirs=[Path('binance_spot_hourly')],
    aug_config=cfg,
    cache_path=Path('$CACHE_PATH'),
    num_workers=16,
)
print(f'Cache built: {len(result)} series', flush=True)
"
    fi
fi

# --- Build training args ---
ARGS=(
    --cache-path       "$CACHE_PATH"
    --output-dir       "$OUTPUT_DIR"
    --num-steps        $NUM_STEPS
    --batch-size       $BATCH_SIZE
    --context-length   $CONTEXT
    --learning-rate    $LR
    --finetune-mode    $FINETUNE_MODE
    --lora-r           $LORA_R
    --torch-dtype      bfloat16
    --r2-prefix        "$R2_PREFIX/finetuned-ckpt"
    --num-workers      16
    --seed             $SEED
    --grad-accum       $GRAD_ACCUM
    --amp-log-std            $AMP_LOG_STD
    --noise-frac             $NOISE_FRAC
    --dropout-rate           $DROPOUT_RATE
    --channel-dropout-prob   $CHANNEL_DROPOUT
    --time-warp-prob         $TIME_WARP
)
if $USE_MUON; then ARGS+=(--use-muon); fi
if [[ -n "$WANDB_PROJECT" ]]; then
    ARGS+=(--wandb-project "$WANDB_PROJECT")
    [[ -n "$WANDB_RUN" ]] && ARGS+=(--wandb-run-name "${WANDB_RUN:-chronos2_${TAG}}")
fi

# --- Run training ---
echo "Starting training at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python chronos2_full_finetune.py "${ARGS[@]}"
TRAIN_STATUS=$?
echo "Training finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit $TRAIN_STATUS)"

if [[ "$TRAIN_STATUS" != "0" ]]; then
    echo "ERROR: training failed (exit $TRAIN_STATUS)"
    exit $TRAIN_STATUS
fi

CKPT="${OUTPUT_DIR}/finetuned-ckpt"

# --- Run linear calibration (global + per-symbol) ---
echo ""
echo "Running buy/sell threshold calibration (global + per-symbol)..."
python chronos2_linear_calibration.py \
    --model-id       "$CKPT" \
    --cal-data-dir   trainingdata \
    --output-path    "${CKPT}/calibration.json" \
    --max-shift-bps  20 \
    --min-gap-bps    2 \
    --grid-steps     25 \
    --max-windows    5000 \
    --per-symbol \
    --hyperparams-dir "hyperparams/chronos2_${TAG}"
echo "Calibration done: ${CKPT}/calibration.json"

# --- Upload calibration to R2 ---
if [[ -n "${R2_ENDPOINT:-}" && -n "${R2_ACCESS_KEY:-}" ]]; then
    echo "Uploading calibration + summary to R2..."
    python - <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
from src.r2_client import R2Client
from pathlib import Path
ckpt = os.environ.get('CKPT_PATH', '')
r2_prefix = os.environ.get('R2_PREFIX', '')
client = R2Client()
for fname in ['calibration.json', 'calibration_short.json']:
    f = Path(ckpt) / fname
    if f.exists():
        client.upload_file(str(f), f'{r2_prefix}/finetuned-ckpt/{fname}')
        print(f'Uploaded {fname} to R2')
summary = Path(ckpt).parent / 'summary.json'
if summary.exists():
    client.upload_file(str(summary), f'{r2_prefix}/summary.json')
    print('Uploaded summary.json to R2')
PYEOF
fi

# --- Print summary ---
echo ""
echo "=== Training complete ==="
python - <<PYEOF
import json
from pathlib import Path
s = Path('${OUTPUT_DIR}/summary.json')
if s.exists():
    d = json.loads(s.read_text())
    b = d.get('baseline_mae_pct', 0)
    f = d.get('finetuned_mae_pct', 0)
    print(f'  Baseline MAE%: {b:.3f}%')
    print(f'  Finetuned MAE%: {f:.3f}%')
    delta = b - f
    print(f'  Improvement: +{delta:.3f}pp ({delta/b*100:+.1f}% relative)')
    print(f'  Checkpoint: ${CKPT}')
    print(f'  R2: ${R2_PREFIX}')
PYEOF
