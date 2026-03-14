#!/bin/bash
# Deploy the unified RL+Gemini trading bot and schedule monitoring checks.
#
# Usage:
#   bash unified_orchestrator/deploy_and_monitor.sh [--live]
#
# This script:
#   1. Starts the unified orchestrator in the background
#   2. Schedules Claude monitoring checks at 1h and 6h
#   3. Re-simulates on latest validation data

set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODE="--dry-run"
if [[ "$1" == "--live" ]]; then
    MODE="--live"
    echo "!!! LIVE TRADING MODE !!!"
    sleep 3
fi

# Activate venv
source .venv313/bin/activate

# --- 1. Start the orchestrator ---
echo "Starting unified orchestrator..."
nohup python -u -m unified_orchestrator.orchestrator \
    --crypto-symbols BTCUSD ETHUSD SOLUSD \
    --model gemini-3.1-flash-lite-preview \
    --thinking-level HIGH \
    $MODE \
    > logs/unified_orchestrator.log 2>&1 &
ORCH_PID=$!
echo "Orchestrator PID: $ORCH_PID"

# --- 2. Schedule monitoring checks ---
# 1-hour check
echo "Scheduling 1-hour monitoring check..."
(sleep 3600 && claude --dangerously-skip-permissions -p "
Check on the unified RL+Gemini trading bot running on this machine.

1. Check if the orchestrator process (PID $ORCH_PID) is still running
2. Read the last 50 lines of logs/unified_orchestrator.log
3. Check strategy_state/unified_state.json for current positions
4. Check strategy_state/fill_events.jsonl for recent fills
5. Verify no errors or crashes
6. Report: is it trading as expected? Any orders placed? Any fills?
7. Run a quick validation: python -m unified_orchestrator.ab_test_rl_gemini --data-path pufferlib_market/data/crypto12_data.bin --days 7

Summarize the 1-hour health check results.
" > logs/monitor_1h.log 2>&1) &
echo "1h check scheduled"

# 6-hour check
echo "Scheduling 6-hour monitoring check..."
(sleep 21600 && claude --dangerously-skip-permissions -p "
Deep health check of the unified RL+Gemini trading bot after 6 hours of running.

1. Check orchestrator process health (PID $ORCH_PID)
2. Read ALL of logs/unified_orchestrator.log - look for patterns:
   - How many cycles completed?
   - Any errors or repeated failures?
   - Were signals generated each cycle?
   - Did any orders fill?
3. Check strategy_state/ for current state
4. If $MODE is --live:
   - Check actual Binance positions vs expected
   - Verify fill events match real orders
5. Re-simulate on latest data to validate performance hasn't degraded:
   python -m unified_orchestrator.ab_test_rl_gemini --data-path pufferlib_market/data/crypto12_data.bin --days 30
6. Check the training runs:
   - tail -5 pufferlib_market/training_crypto12_improved_100M.log
   - tail -5 pufferlib_market/training_stocks10_improved_100M.log
7. Report overall system health and any recommended actions.

Write a detailed 6-hour health report.
" > logs/monitor_6h.log 2>&1) &
echo "6h check scheduled"

echo ""
echo "=== Deployment Summary ==="
echo "  Orchestrator PID: $ORCH_PID"
echo "  Mode: $MODE"
echo "  Log: logs/unified_orchestrator.log"
echo "  1h monitor: logs/monitor_1h.log (at $(date -d '+1 hour' '+%H:%M' 2>/dev/null || date -v+1H '+%H:%M'))"
echo "  6h monitor: logs/monitor_6h.log (at $(date -d '+6 hours' '+%H:%M' 2>/dev/null || date -v+6H '+%H:%M'))"
echo ""
echo "To check progress: tail -f logs/unified_orchestrator.log"
echo "To stop: kill $ORCH_PID"
