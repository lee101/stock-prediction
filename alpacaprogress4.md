# Alpaca Progress 4

## 2026-03-08 Robust Meta Validation + Deploy Update

### Goal
- keep pushing the stock meta-selector, but only promote settings that stay positive on multiple holdout windows and under more realistic execution assumptions.

## 1) Infrastructure changes completed

### Chronos2 / model path
- Fixed the CUDA transformer kernel failure by adding an attention-backend fallback in `binanceneural/model.py`.
- Added stock-hourly known-future time covariates and Chronos2 batch/covariate routing in `binanceneural/forecasts.py`.

### Validation metric
- Extended market-sim scoring to include:
  - return
  - annualized return
  - Sortino
  - max drawdown
  - P&L smoothness
  - ulcer index
  - trade rate
- Wired composite `goodness_score` through:
  - `src/robust_trading_metrics.py`
  - `unified_hourly_experiment/marketsimulator/portfolio_simulator.py`
  - `unified_hourly_experiment/meta_selector.py`
  - `unified_hourly_experiment/sweep_meta_portfolio.py`
  - `unified_hourly_experiment/auto_meta_optimize.py`

### Execution-robust validation
- Added multi-scenario execution validation to `sweep_meta_portfolio.py`:
  - multiple `bar_margin` values in one run
  - multiple `entry_order_ttl_hours` values in one run
  - conservative aggregation uses worst-case scenario metrics for ranking
- Threaded those validation knobs through `auto_meta_optimize.py`.

### Tests
- Validation slice:
  - `python -m pytest tests/test_sweep_meta_portfolio.py tests/test_auto_meta_optimize.py tests/test_meta_selector.py tests/test_portfolio_simulator_goodness.py tests/test_robust_trading_metrics.py -q`
  - result: `30 passed`

## 2) Revalidation results

### A. Current live-style market-entry config

Artifact:
- `experiments/meta_live7_current_robust_20260308.json`

Config revalidated:
- `metric=p10`
- `selection_mode=sticky`
- `switch_margin=0.005`
- `lookback_days=14`
- `sit_out_threshold=-0.001`
- `market_order_entry=true`
- execution scenarios:
  - `bar_margin in {0.0005, 0.0013}`
  - `entry_order_ttl_hours in {0, 1}`

Result:
- `min_sortino=0.1025`
- `mean_sortino=0.3297`
- `min_return_pct=+0.0213`
- `mean_return_pct=+0.1056`
- `mean_max_drawdown_pct=0.9787`
- `min_goodness_score=0.3012`
- `mean_goodness_score=0.5462`
- `min_num_buys=5`

Interpretation:
- positive on `14d/21d/28d`
- still deployable
- but not especially strong

### B. Same live-style selector under limit-entry robustness

Artifact:
- `experiments/meta_live7_current_limit_robust_20260308.json`

Same selector settings, but `market_order_entry=false`.

Result:
- `min_sortino=-1.5087`
- `mean_sortino=-0.9046`
- `min_return_pct=-0.7712`
- `mean_return_pct=-0.5611`
- `mean_max_drawdown_pct=1.3672`
- `min_goodness_score=-1.9100`
- `mean_goodness_score=-1.1947`
- `min_num_buys=3`

Execution sensitivity:
- `13bp` buffer is materially worse than `5bp`
- TTL `0` vs `1` made little difference here

Interpretation:
- current selector is not robust if we require passive/limit-style fills

### C. Old positive limit-style recipe no longer holds

Artifact:
- `experiments/meta_top5_limit_robust_20260308.json`

Revalidated prior March 5 candidate:
- five `wd*` strategies
- `metric=sharpe`
- `selection_mode=winner`
- `lookback_days=16`
- `sit_out_threshold=0.3`
- `edge=0.0065`
- `market_order_entry=false`

Result:
- `min_sortino=-1.9268`
- `mean_sortino=-1.6150`
- `min_return_pct=-1.5422`
- `mean_return_pct=-0.8560`
- `mean_max_drawdown_pct=1.3923`
- `min_goodness_score=-3.0475`

Interpretation:
- do not resurrect this older limit-style deployment

## 3) New market-entry refine around the live regime

Artifact:
- `experiments/meta_live7_market_refine_20260308.json`

Search space:
- `metric=p10`
- `selection_mode in {winner, sticky}`
- `switch_margin in {0.0, 0.005}`
- `lookback_days in {10, 14, 16}`
- `sit_out_threshold in {-0.001, 0.0, 0.25, 0.3}`
- `market_order_entry=true`
- same 7 live strategies
- execution validation:
  - `bar_margin in {0.0005, 0.0013}`
  - `entry_order_ttl_hours in {0, 1}`

Raw top row:
- `sticky`
- `switch_margin=0.005`
- `lookback_days=14`
- `sit_out_threshold=0.0`
- `min_goodness_score=2.1535`
- `mean_goodness_score=2.3182`
- `min_num_buys=1`

Decision on raw top row:
- rejected for deploy because activity is too sparse (`min_num_buys=1`)

Best activity-filtered candidate (`min_num_buys >= 2`):
- `sticky`
- `switch_margin=0.005`
- `lookback_days=16`
- `sit_out_threshold=-0.001`
- `min_sortino=1.7241`
- `mean_sortino=2.6332`
- `min_return_pct=+0.2968`
- `mean_return_pct=+0.4847`
- `min_annualized_return_pct=+4.27`
- `mean_annualized_return_pct=+6.57`
- `mean_max_drawdown_pct=0.3413`
- `min_goodness_score=2.0194`
- `mean_goodness_score=2.9241`
- `min_num_buys=2`
- `mean_num_buys=4.67`

Comparison vs current live-style market-entry baseline:
- current: `min_goodness_score=0.3012`, `mean_goodness_score=0.5462`
- refined activity-filtered candidate: `min_goodness_score=2.0194`, `mean_goodness_score=2.9241`

Interpretation:
- this is a real improvement while still keeping some activity on every holdout
- still market-entry based, not limit-fill robust
- annualization is now printed in sweep outputs using the simulator frequency basis

Annualized confirmation artifact:
- `experiments/meta_live7_activity_filtered_ann_20260308.json`

## 4) Deploy decision

### Promoted in checked-in supervisor config
- Updated `supervisor/unified-stock-trader.conf`
- change:
  - `--meta-lookback-days 14` -> `--meta-lookback-days 16`
- kept:
  - `metric=p10`
  - `selection_mode=sticky`
  - `switch_margin=0.005`
  - `sit_out_threshold=-0.001`
  - `market_order_entry`
  - `short_intensity_multiplier=1.5`

### Not promoted
- no limit-entry candidate
- no old March 5 top-five recipe
- no zero-trade / one-trade sparse selector rows

## 5) Practical conclusion

- Best currently validated deploy path is still the market-entry meta selector.
- Best checked-in update is the activity-filtered refine:
  - `sticky p10`
  - `lookback=16`
  - `switch_margin=0.005`
  - `sit_out=-0.001`
- The repo now has the machinery to score candidates on execution robustness directly.
- Next work should be to find a truly positive limit-entry configuration that survives `5bp` to `13bp` fill assumptions, because the current live family does not.

## 2026-03-08 Crypto Chronos selector work

### Code changes completed
- Added explicit recent-history control for the Binance selector data path:
  - `DatasetConfig.max_history_hours`
  - forecast windowing + tail-read support in `binanceexp1/data.py`
  - CLI plumbing through:
    - `binanceexp1/train_multiasset_selector_robust.py`
    - `binanceexp1/search_checkpoint_sets_robust.py`
    - `binanceexp1/run_multiasset_selector.py`
    - `binanceexp1/sweep_chronos_feature_configs_robust.py`
- Added the same recent-history limiter to grouped Chronos cache generation in `binanceexp1/joint_chronos_forecast_cache.py`.
- Fixed a reproducibility bug in the Chronos sweep:
  - sweep runs previously used a generic `chronos_feature_sweep` prefix
  - interrupted / older checkpoint directories could be silently reused across different feature experiments
  - the sweep now derives a unique sanitized run prefix from `feature_experiment_name`

### Validation
- Targeted tests:
  - `pytest tests/test_binance_sweep_chronos_feature_configs_robust.py tests/test_binance_train_multiasset_selector_robust.py tests/test_binance_joint_chronos_forecast_cache.py tests/test_binanceexp1_data_chronos_env_overrides.py`
  - result: `13 passed`

### Experiment status
- Grouped `1h/6h/24h` cross-learning run with:
  - `max_history_hours=2880`
  - grouped cache rebuild
  - realistic selector search
  - full training (`dry_train_steps=0`)
- Result:
  - still too slow to treat as a practical sweep at the current `1024` context / `1h,6h,24h` setup
  - the run was stopped after confirming it was spending several minutes in grouped cache generation before producing the first cache artifact
- Standard per-symbol `1h/6h/24h` path:
  - seeded existing `1h/24h` caches successfully
  - generated missing `6h` cache for `BTCUSD`
  - existing seeded `6h` cache for `ETHUSD` was reused
  - `SOLUSD` `6h` cache remained the slow missing leg
- Clean full-training rerun attempt exposed the checkpoint-prefix collision bug above; after fixing it, a shortened clean rerun was started but not completed in this session

### Current conclusion
- The infra is materially better:
  - recent-history capped Chronos cache generation
  - grouped cache limiter
  - no more feature-sweep checkpoint-prefix collisions
- I do not have a clean completed new `1h/6h/24h` ranking yet from this session.
- Do not deploy any new crypto selector change from this work yet.
