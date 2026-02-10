# Alpaca Selector Decision Lag Sweep - 2026-02-10

Goal: evaluate the hourly selector with a **1-bar decision lag** (`decision_lag_bars=1`) to better match
live execution (action computed on bar *t* is executed on bar *t+1*).

This sweeps selector thresholds (intensity, `min_edge`, `risk_weight`) on a fixed checkpoint over a short
evaluation window, writing a ranked CSV.

## Run

```bash
source .venv/bin/activate

python experiments/alpaca_selector_lag1_tune_20260210/run_sweep.py
```

Output: `experiments/alpaca_selector_lag1_tune_20260210/sweep_results.csv`

