#!/bin/bash
# DISABLED - Moving to hourly trading
# Daily workflow: Update data, retrain on all data, generate trading signals
# Run this via cron every day before market open
# This script is currently commented out and not in use

exit 0  # Script disabled

set -e

LOGFILE="daily_workflow_$(date +%Y%m%d).log"

echo "======================================" | tee -a $LOGFILE
echo "Daily Neural Trading Workflow" | tee -a $LOGFILE
echo "Started: $(date)" | tee -a $LOGFILE
echo "======================================" | tee -a $LOGFILE

cd /nvme0n1-disk/code/stock-prediction

# Activate environment
source .venv313/bin/activate

# Step 1: Update market data to latest
echo "" | tee -a $LOGFILE
echo "Step 1: Updating market data..." | tee -a $LOGFILE
python update_daily_data.py 2>&1 | tee -a $LOGFILE

# Step 2: Update Chronos forecasts for key symbols
echo "" | tee -a $LOGFILE
echo "Step 2: Updating Chronos forecasts..." | tee -a $LOGFILE
python update_key_forecasts.py 2>&1 | tee -a $LOGFILE

# Step 3: Retrain model on ALL data (Phase 2 - validation_days=0)
echo "" | tee -a $LOGFILE
echo "Step 3: Retraining model on latest data..." | tee -a $LOGFILE
python final_fit_all_data.py --output-name daily_$(date +%Y%m%d) 2>&1 | tee -a $LOGFILE

# Step 4: Generate trading signals (dry run)
echo "" | tee -a $LOGFILE
echo "Step 4: Generating trading signals..." | tee -a $LOGFILE
python daily_trading_with_neural.py 2>&1 | tee -a $LOGFILE

# Step 5: Optionally run simulation to verify performance
echo "" | tee -a $LOGFILE
echo "Step 5: Running performance verification..." | tee -a $LOGFILE
LATEST_CHECKPOINT=$(find neuraldailytraining/checkpoints/final_daily_* -name "epoch_*.pt" | sort | tail -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    PYTHONPATH=. python neuraldailymarketsimulator/simulator.py \
        --checkpoint "$LATEST_CHECKPOINT" \
        --days 10 \
        --start-date 2025-10-05 2>&1 | tee -a $LOGFILE
fi

echo "" | tee -a $LOGFILE
echo "======================================" | tee -a $LOGFILE
echo "Daily workflow completed: $(date)" | tee -a $LOGFILE
echo "======================================" | tee -a $LOGFILE
