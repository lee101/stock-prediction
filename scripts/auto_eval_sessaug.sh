#!/usr/bin/env bash
# Auto-evaluate sessaug checkpoints every 30 minutes until training finishes.
# Run: nohup bash scripts/auto_eval_sessaug.sh &
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source .venv313/bin/activate

LOG="strategy_state/crypto30_daily/auto_eval.log"
mkdir -p "$(dirname "$LOG")"

echo "$(date -u +%FT%TZ) auto_eval started" >> "$LOG"

while true; do
    # Check if any training is still running
    n_running=$(ps aux | grep "pufferlib_market.train" | grep crypto30 | grep -v grep | wc -l)
    if [ "$n_running" -eq 0 ]; then
        echo "$(date -u +%FT%TZ) No training running, final eval" >> "$LOG"
        python scripts/eval_crypto30_sessaug.py 2>&1 | tee -a "$LOG"
        echo "$(date -u +%FT%TZ) auto_eval done" >> "$LOG"
        break
    fi

    echo "$(date -u +%FT%TZ) $n_running training jobs active, running eval" >> "$LOG"
    python scripts/eval_crypto30_sessaug.py 2>&1 | tee -a "$LOG"
    echo "---" >> "$LOG"

    sleep 1800  # 30 minutes
done
