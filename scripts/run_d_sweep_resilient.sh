#!/bin/bash
# Resilient D sweep wrapper - restarts if killed
SEEDS_LIST="$1"
LOG="/nvme0n1-disk/code/stock-prediction/.d_sweep_resilient_$(date +%s).log"
cd /nvme0n1-disk/code/stock-prediction
while true; do
    SEEDS="$SEEDS_LIST" bash scripts/sweep_screened32.sh D >> "$LOG" 2>&1
    echo "[wrapper] D sweep exited at $(date -u +%FT%TZ), restarting..." >> "$LOG"
    sleep 3
done
