#!/bin/bash
# Example usage of batch_run_market_simulator.py

echo "==================================================="
echo "Example 1: Run with default symbols (from trade_stock_e2e.py)"
echo "==================================================="
# Uncomment to run:
# python batch_run_market_simulator.py --limit 3 --simulation-days 10

echo ""
echo "==================================================="
echo "Example 2: Test with specific symbols"
echo "==================================================="
# Uncomment to run:
# python batch_run_market_simulator.py --symbols AAPL MSFT NVDA --simulation-days 30

echo ""
echo "==================================================="
echo "Example 3: Download data only (10 years)"
echo "==================================================="
# Uncomment to run:
# python batch_run_market_simulator.py --download-only --data-years 10

echo ""
echo "==================================================="
echo "Example 4: Run simulation with existing data"
echo "==================================================="
# Uncomment to run:
# python batch_run_market_simulator.py --skip-download --simulation-days 60

echo ""
echo "==================================================="
echo "Example 5: Get all Alpaca tradable stocks (limited)"
echo "==================================================="
# Uncomment to run:
# python batch_run_market_simulator.py --use-all-alpaca --asset-class us_equity --limit 20

echo ""
echo "==================================================="
echo "Example 6: Full production run"
echo "==================================================="
# Uncomment to run:
# python batch_run_market_simulator.py \
#     --simulation-days 60 \
#     --data-years 10 \
#     --initial-cash 100000 \
#     --run-name production_backtest_$(date +%Y%m%d)

echo ""
echo "Uncomment the examples above to run them."
echo "Results will be saved to: strategytraining/batch_results/"
echo "Training data will be saved to: trainingdata/"
