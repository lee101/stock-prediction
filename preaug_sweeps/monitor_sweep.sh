#!/bin/bash
# Monitor the pre-augmentation sweep progress

LOGFILE=$(ls -t preaug_sweeps/logs/full_sweep_*.log 2>/dev/null | head -1)

if [ -z "$LOGFILE" ]; then
    echo "No sweep log found!"
    exit 1
fi

echo "=============================================================================="
echo "PRE-AUGMENTATION SWEEP MONITOR"
echo "=============================================================================="
echo "Log file: $LOGFILE"
echo ""

# Check if sweep is still running
PID=$(pgrep -f "sweep_runner.py.*ETHUSD.*UNIUSD.*BTCUSD")
if [ -n "$PID" ]; then
    echo "✓ Sweep is RUNNING (PID: $PID)"
else
    echo "✗ Sweep is NOT running"
fi
echo ""

# Show progress
echo "=============================================================================="
echo "PROGRESS"
echo "=============================================================================="
echo ""

# Count completed strategies
COMPLETED=$(grep -c "✓ Completed" "$LOGFILE" 2>/dev/null || echo "0")
FAILED=$(grep -c "✗ Failed" "$LOGFILE" 2>/dev/null || echo "0")
TOTAL=24  # 3 symbols × 8 strategies

echo "Progress: $COMPLETED/$TOTAL completed, $FAILED failed"
echo ""

# Show recent activity
echo "=============================================================================="
echo "RECENT ACTIVITY (last 20 lines)"
echo "=============================================================================="
tail -20 "$LOGFILE" | grep -E "(INFO|Epoch|Testing|✓|✗|MAE|Completed|Failed)" || tail -20 "$LOGFILE"
echo ""

# Show results so far
echo "=============================================================================="
echo "RESULTS SO FAR"
echo "=============================================================================="
grep "✓ Completed" "$LOGFILE" 2>/dev/null | tail -10 || echo "No results yet..."
echo ""

# Show failed runs
if [ "$FAILED" -gt 0 ]; then
    echo "=============================================================================="
    echo "FAILED RUNS"
    echo "=============================================================================="
    grep "✗ Failed" "$LOGFILE" | tail -10
    echo ""
fi

# Estimated time remaining
if [ "$COMPLETED" -gt 0 ]; then
    REMAINING=$((TOTAL - COMPLETED))
    echo "Estimated: ~$REMAINING more runs to complete"
fi

echo ""
echo "=============================================================================="
echo "To watch live: tail -f $LOGFILE"
echo "To re-run monitor: ./preaug_sweeps/monitor_sweep.sh"
echo "=============================================================================="
