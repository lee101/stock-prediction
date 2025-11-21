#!/bin/bash
# DISABLED - Moving to hourly trading
# Automatically install cron jobs for neural trading automation
# This script is currently commented out and not in use

exit 0  # Script disabled

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Neural Trading Cron Jobs"
echo "===================================="
echo ""

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Get current crontab
crontab -l > /tmp/current_cron 2>/dev/null || touch /tmp/current_cron

# Check if jobs already exist
if grep -q "daily_retrain_and_trade.sh" /tmp/current_cron; then
    echo "⚠️  Daily workflow job already exists in crontab"
else
    echo "# Neural Trading - Daily workflow (6 AM)" >> /tmp/current_cron
    echo "0 6 * * * cd $SCRIPT_DIR && ./daily_retrain_and_trade.sh >> logs/daily_workflow.log 2>&1" >> /tmp/current_cron
    echo "✅ Added daily workflow job"
fi

if grep -q "launch_training.sh phase1" /tmp/current_cron; then
    echo "⚠️  Weekly Phase 1 job already exists in crontab"
else
    echo "# Neural Trading - Weekly Phase 1 search (Sunday 2 AM)" >> /tmp/current_cron
    echo "0 2 * * 0 cd $SCRIPT_DIR && ./launch_training.sh phase1 >> logs/weekly_phase1.log 2>&1" >> /tmp/current_cron
    echo "✅ Added weekly Phase 1 search job"
fi

# Install new crontab
crontab /tmp/current_cron

echo ""
echo "✅ Cron jobs installed successfully!"
echo ""
echo "Current cron schedule:"
echo "====================="
crontab -l | grep -A1 "Neural Trading"
echo ""
echo "To view all cron jobs: crontab -l"
echo "To edit cron jobs: crontab -e"
echo "To remove all jobs: crontab -r"
echo ""
echo "Logs will be written to:"
echo "  $SCRIPT_DIR/logs/daily_workflow.log"
echo "  $SCRIPT_DIR/logs/weekly_phase1.log"
