Boost Baseline (XGBoost/SKLearn) over forecasts

Overview
- Builds a lightweight dataset from cached `results/predictions-*.csv` rows for a symbol (e.g., ETHUSD).
- Joins those snapshots to `trainingdata/train/<SYMBOL>.csv` to compute realized next-day returns.
- Trains a boosted regressor (XGBoost if available, else scikit-learn GradientBoostingRegressor) to predict next-day return from the forecast features (predicted deltas, losses, profits).
- Runs a simple backtest to pick position-sizing scale and cap, with basic fee modeling. Outputs baseline metrics and a suggested position size for the most recent forecast.

Quick Start
- Ensure you have historical price CSV under `trainingdata/train/ETHUSD.csv` and cached prediction snapshots under `results/predictions-*.csv` that include `instrument == ETHUSD`.
- Run:
  - `PYTHONPATH=$(pwd) .env/bin/python -m boostbaseline.run_baseline ETHUSD`

What it does
- Gathers features for each snapshot:
  - Predicted vs last price deltas for close/high/low
  - Validation losses (close/high/low)
  - Profit metrics when present (takeprofit/maxdiffprofit/entry_takeprofit)
- Targets are next-day close-to-close returns from `trainingdata` aligned to snapshot time.
- Trains regressor → predicts returns → selects scale `k` and cap `c` by backtest grid to maximize compounded return with fees.

Artifacts
- Saves model under `boostbaseline/models/<symbol>_boost.model` (XGB JSON or SKLearn joblib).
- Writes a short report to `baselineperf.md` and prints summary.

Notes
- If `xgboost` is not installed, the code falls back to `sklearn.ensemble.GradientBoostingRegressor` which is already in `requirements.txt`.
- Fee model is simple and conservative; refine in `boostbaseline/backtest.py` if needed.

