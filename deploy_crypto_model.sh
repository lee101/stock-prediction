#!/usr/bin/env bash
set -euo pipefail

LAUNCH=/home/lee/code/stock/deployments/binance-hybrid-spot/launch.sh
LOG=/var/log/supervisor/binance-hybrid-spot.log
SUPERVISOR_PROG=binance-hybrid-spot

usage() {
    cat <<'USAGE'
Usage: deploy_crypto_model.sh [--dry-run] [--remove-rl] <checkpoint_path> [symbols...]

Deploy an RL checkpoint to the Binance hybrid spot bot.

  checkpoint_path   Path to .pt checkpoint file (absolute or relative)
  symbols           Override symbols (default: keep existing)

Options:
  --dry-run         Show what would change without modifying anything
  --remove-rl       Remove --rl-checkpoint from launch.sh (revert to LLM-only)

Examples:
  deploy_crypto_model.sh rl-trainingbinance/checkpoints/best.pt
  deploy_crypto_model.sh --dry-run binanceneural/checkpoints/epoch_003.pt
  deploy_crypto_model.sh --remove-rl
USAGE
    exit 0
}

DRY_RUN=0
REMOVE_RL=0
CHECKPOINT=""
SYMBOLS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --dry-run) DRY_RUN=1; shift ;;
        --remove-rl) REMOVE_RL=1; shift ;;
        -*) echo "ERROR: Unknown option: $1"; exit 1 ;;
        *)
            if [[ -z "$CHECKPOINT" ]]; then
                CHECKPOINT="$1"
            else
                SYMBOLS="${SYMBOLS:+$SYMBOLS }$1"
            fi
            shift
            ;;
    esac
done

if [[ "$REMOVE_RL" -eq 1 ]]; then
    echo "Removing --rl-checkpoint from $LAUNCH"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] Would remove --rl-checkpoint line from launch.sh"
        exit 0
    fi
    cp "$LAUNCH" "${LAUNCH}.bak.$(date +%s)"
    sed -i '/--rl-checkpoint/d' "$LAUNCH"
    echo "Removed --rl-checkpoint from launch.sh"
    echo ""
    echo "Restarting $SUPERVISOR_PROG..."
    echo "ilu" | sudo -S supervisorctl restart "$SUPERVISOR_PROG" 2>/dev/null
    echo "Done."
    exit 0
fi

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint_path required (or use --remove-rl)"
    echo "Run with --help for usage."
    exit 1
fi

if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="/home/lee/code/stock/$CHECKPOINT"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"

if [[ ! -f "$LAUNCH" ]]; then
    echo "ERROR: Launch script not found: $LAUNCH"
    exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    if grep -q '\-\-rl-checkpoint' "$LAUNCH"; then
        echo "[dry-run] Would update existing --rl-checkpoint to: $CHECKPOINT"
    else
        echo "[dry-run] Would add --rl-checkpoint $CHECKPOINT to launch.sh"
    fi
    [[ -n "$SYMBOLS" ]] && echo "[dry-run] Would update --symbols to: $SYMBOLS"
    echo "[dry-run] No changes applied."
    exit 0
fi

backup="${LAUNCH}.bak.$(date +%s)"
cp "$LAUNCH" "$backup"
echo "Backup: $backup"

if grep -q '\-\-rl-checkpoint' "$LAUNCH"; then
    sed -i "s|--rl-checkpoint [^ \\\\]*|--rl-checkpoint $CHECKPOINT|" "$LAUNCH"
    echo "Updated --rl-checkpoint in launch.sh"
else
    sed -i "/--fallback-mode chronos2/a\\  --rl-checkpoint $CHECKPOINT \\\\" "$LAUNCH"
    echo "Added --rl-checkpoint to launch.sh"
fi

if [[ -n "$SYMBOLS" ]]; then
    sed -i "s|--symbols [^\\\\]*\\\\|--symbols $SYMBOLS \\\\|" "$LAUNCH"
    echo "Updated --symbols to: $SYMBOLS"
fi

echo ""
echo "--- Updated launch.sh ---"
cat "$LAUNCH"
echo "--- end ---"

echo ""
echo "Restarting $SUPERVISOR_PROG..."
echo "ilu" | sudo -S supervisorctl restart "$SUPERVISOR_PROG" 2>/dev/null

sleep 5
if [[ -f "$LOG" ]]; then
    echo ""
    echo "--- Recent logs ---"
    tail -20 "$LOG"
fi

echo ""
echo "Deployment complete."
