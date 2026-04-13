#!/bin/bash
# Continuous monitor loop: runs monitor_sweeps.sh every 30 minutes
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

echo "[$(date -u +%FT%TZ)] Monitor loop starting..."

while true; do
  bash scripts/monitor_sweeps.sh
  echo "[$(date -u +%FT%TZ)] Sleeping 30m until next monitor run..."
  sleep 1800
done
