#!/bin/bash
# Run all remote experiments sequentially
set -e
cd /home/administrator/code/stock
source .venv/bin/activate

echo "=== Phase 1: Architecture sweep ==="
python unified_hourly_experiment/run_full_experiment.py \
    --configs exp_1024h_4L exp_768h_4L_stockonly exp_512h_4L_baseline \
    2>&1 | tee /tmp/remote_phase1.log

echo "=== Phase 2: Focused stock subsets ==="
python unified_hourly_experiment/run_focused_experiment.py \
    --configs focused_4_200ep focused_8_200ep longable_200ep shortable_200ep \
    2>&1 | tee /tmp/remote_phase2.log

echo "=== Done ==="
