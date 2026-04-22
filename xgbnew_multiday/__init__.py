"""Multi-day / week-scale XGBoost stock trader prototype.

Research-only companion to ``xgbnew/`` which only does single-day open-to-close
trades. The central question: can a model predict WHEN to hold 1 day vs 3 days
vs 10 days?

Approach (option-a from the task spec):
  * Train one binary XGB per horizon N in {1, 2, 3, 5, 10} predicting
    ``forward_N_day_return > 0``.
  * Meta-selector picks the best horizon per (symbol, day) using expected
    return proxy: ``E[r_N] ≈ (p_N - 0.5) * hist_abs_ret_N``.
  * Optional: constrain to a single horizon per day (fastest path to test).

Deliberately kept separate from ``xgbnew`` so experiments here cannot affect
``xgb-daily-trader-live``.
"""
