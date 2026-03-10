#!/bin/bash
# Run a batch of LLM trader experiments, staying within daily API quota.
# Designed to be called by cron each day at 07:15 UTC (midnight Pacific + 15m buffer).
# Uses disk cache so interrupted/resumed runs don't repeat calls.

set -e
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

LOG_DIR="llm_hourly_trader/logs"
mkdir -p "$LOG_DIR"
DATE=$(date -u +%Y%m%d)
LOG="$LOG_DIR/batch_${DATE}.log"

echo "=== LLM Hourly Trader Batch - $(date -u) ===" >> "$LOG"

# Each run uses disk cache, so we can run the same command daily
# and it picks up where it left off.
# Budget: ~450 calls/day (leave 50 buffer)

# Priority 1: BTC 7d with gemini-2.5-flash (151 remaining from cache)
python -m llm_hourly_trader.backtest \
    --symbols BTCUSD --days 7 --prompt default \
    --model gemini-2.5-flash --rate-limit 7.0 \
    >> "$LOG" 2>&1 || true

# Priority 2: BTC+ETH 7d with gemini-2.5-flash
python -m llm_hourly_trader.backtest \
    --symbols BTCUSD ETHUSD --days 7 --prompt default \
    --model gemini-2.5-flash --rate-limit 7.0 \
    >> "$LOG" 2>&1 || true

# Priority 3: Full crypto 7d default
python -m llm_hourly_trader.backtest \
    --group crypto --days 7 --prompt default \
    --model gemini-2.5-flash --rate-limit 7.0 \
    >> "$LOG" 2>&1 || true

# Priority 4: Crypto 7d conservative
python -m llm_hourly_trader.backtest \
    --group crypto --days 7 --prompt conservative \
    --model gemini-2.5-flash --rate-limit 7.0 \
    >> "$LOG" 2>&1 || true

echo "=== Batch complete - $(date -u) ===" >> "$LOG"
