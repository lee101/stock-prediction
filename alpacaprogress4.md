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

## 2026-03-08 Limit-Entry First-Trigger Recovery

### Code changes completed
- Added selector-simulation `entry_selection_mode` plumbing through:
  - `unified_hourly_experiment/sweep_meta_portfolio.py`
  - `unified_hourly_experiment/auto_meta_optimize.py`
  - `unified_hourly_experiment/trade_unified_hourly_meta.py`
- Added live CLI support for `meta_metric=goodness` in `trade_unified_hourly_meta.py`.

### Tests
- `pytest -q tests/test_auto_meta_optimize.py tests/test_trade_unified_hourly_meta.py tests/test_sweep_meta_portfolio.py tests/test_meta_selector.py`
- result: `28 passed`

## 1) Limit-entry revalidation with `first_trigger`

Artifact:
- `experiments/meta_live7_limit_firsttrigger_baseline_20260308.json`

Revalidated selector:
- `metric=p10`
- `selection_mode=sticky`
- `lookback_days=16`
- `switch_margin=0.005`
- `sit_out_threshold=-0.001`
- `entry_selection_mode=first_trigger`
- `market_order_entry=false`
- execution scenarios:
  - `bar_margin in {0.0005, 0.0013}`
  - `entry_order_ttl_hours in {0, 1}`

Result:
- `min_sortino=2.3684`
- `mean_sortino=3.5559`
- `min_return_pct=+0.4065`
- `mean_return_pct=+0.6166`
- `min_annualized_return_pct=+5.89`
- `mean_annualized_return_pct=+8.51`
- `mean_max_drawdown_pct=0.3468`
- `min_goodness_score=2.6378`
- `mean_goodness_score=3.7957`
- `min_num_buys=2`

Interpretation:
- the missing knob was not limit-vs-market alone; it was how passive fills are prioritized in selector simulations
- `first_trigger` flips the previously negative limit-entry family strongly positive on `14d/21d/28d`

## 2) Focused refine around the recovered limit-entry family

Artifact:
- `experiments/meta_live7_limit_firsttrigger_sticky_refine_20260308.json`

Search space:
- `metric=p10`
- `selection_mode=sticky`
- `switch_margin in {0.0, 0.0025, 0.005, 0.0075}`
- `recency_halflife_days in {0, 4}`
- `lookback_days in {14, 16, 18, 21}`
- `sit_out_threshold in {-0.001, 0.0, 0.001, 0.0025}`
- `entry_selection_mode=first_trigger`
- `market_order_entry=false`
- same 7 live strategies
- same execution validation scenarios

Best row by conservative goodness ranking:
- `sticky`
- `lookback_days=18`
- `switch_margin=0.0`
- `sit_out_threshold=0.0`
- `min_sortino=2.7132`
- `mean_sortino=4.0950`
- `min_return_pct=+0.4450`
- `mean_return_pct=+0.4562`
- `mean_max_drawdown_pct=0.2134`
- `min_goodness_score=3.0025`
- `mean_goodness_score=4.1079`
- `min_num_buys=2`

Best balanced positive-PnL candidate:
- `sticky`
- `lookback_days=18`
- `switch_margin=0.0025`
- `sit_out_threshold=-0.001`
- equivalent metrics also appeared at `switch_margin=0.005`
- `min_sortino=2.5453`
- `mean_sortino=3.6908`
- `min_return_pct=+0.4139`
- `mean_return_pct=+0.6282`
- `min_annualized_return_pct=+6.15`
- `mean_annualized_return_pct=+8.66`
- `mean_max_drawdown_pct=0.3467`
- `min_goodness_score=2.7965`
- `mean_goodness_score=3.9151`
- `min_num_buys=2`
- `mean_num_buys=4.67`

Interpretation:
- the refine improved over the recovered limit-entry baseline
- the `lookback=18` family is now the frontier
- the `switch=0.0025/0.005, sit_out=-0.001` row is the better deploy candidate if PnL remains the primary objective
- the `switch=0.0, sit_out=0.0` row is the more conservative choice if we want tighter drawdown and higher goodness floor

## 3) Comparison vs current market-entry deploy candidate

Current market-entry candidate from earlier in this note:
- `sticky p10`
- `lookback=16`
- `switch=0.005`
- `sit_out=-0.001`
- `market_order_entry=true`
- `min_return_pct=+0.2968`
- `mean_return_pct=+0.4847`
- `min_goodness_score=2.0194`
- `mean_goodness_score=2.9241`

Refined limit-entry candidate:
- `sticky p10`
- `lookback=18`
- `switch=0.0025` (same metrics at `0.005`)
- `sit_out=-0.001`
- `entry_selection_mode=first_trigger`
- `market_order_entry=false`
- `min_return_pct=+0.4139`
- `mean_return_pct=+0.6282`
- `min_goodness_score=2.7965`
- `mean_goodness_score=3.9151`

Interpretation:
- the recovered limit-entry family now beats the promoted market-entry family on both return and goodness across the validated holdouts

## 4) Deploy decision

### Promoted in checked-in supervisor config
- Updated `supervisor/unified-stock-trader.conf`
- changes:
  - removed `--market-order-entry`
  - added `--entry-selection-mode first_trigger`
  - `--meta-lookback-days 16` -> `18`
  - kept `--meta-switch-margin 0.005`
  - kept `--sit-out-threshold -0.001`

Rationale:
- `switch=0.005` matched the stronger balanced limit-entry row while minimizing deploy delta from the current market-entry config
- the resulting config is now passive-fill aligned and validated positive under both `5bp` and `13bp` fill assumptions

### Not done here
- did not run `supervisorctl` from this repo session
- this turn only updates the checked-in deploy config and artifacts

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

## 2026-03-08 Chronos Feature-Interaction PnL Sweep

### Goal
- test whether richer Chronos integration into the Binance selector improves robust simulated PnL:
  - cross-horizon interaction features on top of `1h/24h`
  - `1h/6h/24h` Chronos features plus the same interaction layer

### Code changes completed
- Added opt-in Chronos interaction features to `binanceexp1/data.py` and `binanceexp1/config.py`:
  - per-horizon `forecast_range_pct_h*`
  - pairwise horizon spreads for delta / confidence / range
  - aggregate curve features:
    - `forecast_delta_mean`
    - `forecast_delta_abs_mean`
    - `forecast_confidence_mean`
    - `forecast_range_mean`
    - `forecast_weighted_delta`
    - `forecast_weighted_agreement`
    - `forecast_curve_slope`
    - `forecast_curve_residual`
- Threaded `--use-forecast-interactions` through:
  - `binanceexp1/train_multiasset_selector_robust.py`
  - `binanceexp1/search_checkpoint_sets_robust.py`
  - `binanceexp1/run_multiasset_selector.py`
  - `binanceexp1/sweep_chronos_feature_configs_robust.py`
- Improved feature-horizon inference in `src/forecast_horizon_utils.py` so saved feature sets that include `forecast_*` and `predicted_*` columns retain their required horizons.
- Updated the Binance runtime loader to prefer checkpoint-saved `feature_columns` when available, which is the safer path for future expanded-feature checkpoints.

### Validation
- Targeted tests:
  - `python -m pytest tests/test_binanceexp1_forecast_interactions.py tests/test_binance_sweep_chronos_feature_configs_robust.py tests/test_binance_train_multiasset_selector_robust.py tests/test_forecast_horizon_utils.py tests/test_binance_joint_chronos_forecast_cache.py tests/test_binanceexp1_data_chronos_env_overrides.py -q`
  - result: `19 passed`

### Experiment artifacts
- Sweep root:
  - `experiments/binance_selector_chronos_pnl_push_fast_20260308/ranking.csv`
- Per-config search outputs:
  - `experiments/binance_selector_chronos_pnl_push_fast_20260308/h1_24_ctx336_interactions/search/ranking.csv`
  - `experiments/binance_selector_chronos_pnl_push_fast_20260308/h1_6_24_ctx512_interactions/search/ranking.csv`

### A. `1h/24h` plus interaction features

Config:
- `h1_24_ctx336_interactions`
- trained with fast real finetune schedule:
  - `epochs=3`
  - `batch_size=32`
  - `max_history_hours=1440`
  - `sequence_length=72`

Best overall combo inside this feature family:
- still the seeded baseline checkpoints, not the newly trained interaction checkpoints
- `selection_score=-0.6623`
- `worst_ret=+0.1887%`
- `mean_ret=+3.6870%`
- `worst_ann=+5.0676%`
- `mean_ann=+205.5387%`
- `trade_count_mean=202.0`

Best combo that actually used a new interaction checkpoint:
- ETH-only replacement
- `selection_score=-15.4507`
- `worst_ret=-0.2698%`
- `mean_ret=+3.2290%`

Interpretation:
- the new interaction checkpoints did not beat the seeded baseline combo
- interaction features on top of `1h/24h` are not a promotion candidate yet

### B. `1h/6h/24h` plus interaction features

Config:
- `h1_6_24_ctx512_interactions`
- same fast real finetune schedule as above

Best overall combo inside this feature family:
- again remained the seeded baseline checkpoints
- `selection_score=-14.4680`
- `worst_ret=-0.0699%`
- `mean_ret=+3.4252%`
- `worst_ann=-1.8165%`
- `mean_ann=+186.0926%`
- `trade_count_mean=200.25`

Best combo that actually used a new checkpoint:
- ETH-only replacement
- `selection_score=-40.2475`
- `worst_ret=-2.6432%`
- `mean_ret=+1.0590%`

Observed runtime / data quality issue:
- during `SOLUSD` `6h` forecast generation:
  - `Used heuristic fallback for 4/64 forecasts for SOLUSD (horizon=6h) due to invalid Chronos outputs.`

Interpretation:
- adding the `6h` leg made the feature family materially worse in robust selection score
- the `6h` forecasts are still not reliable enough for selector promotion in this setup
- runtime is also meaningfully worse because the missing `6h` cache leg had to be generated before training

### Deploy decision
- do not deploy `use_forecast_interactions`
- do not deploy `1h/6h/24h` selector features
- keep the current crypto selector path unchanged

### Next step
- if we continue on this line, the next experiment should isolate forecast quality first:
  - rebuild / validate the `6h` Chronos cache leg per symbol
  - reject or down-weight heuristic-fallback windows
  - only then retry selector retraining with `1h/6h/24h`

## 2026-03-08 Crypto Selector Fill-Realism Upgrade

### Chronos fallback clarification
- `Chronos2` is still the forecasting model.
- In this repo, "Chronos fallback" means a row-level heuristic replacement used only when Chronos emits invalid forecast rows:
  - non-finite or non-positive prices
  - extreme prices relative to recent close
  - `predicted_low_p50 > predicted_high_p50`
- Relevant code paths:
  - `binanceneural/forecasts.py`
  - `binanceexp1/joint_chronos_forecast_cache.py`
- This is a safety rail for bad forecast rows, not a deliberate switch away from Chronos2.

### Market-sim realism change completed
- Added an optional penetration-aware passive-fill model in:
  - `newnanoalpacahourlyexp/marketsimulator/selector.py`
- New selector knobs:
  - `limit_fill_model in {binary, penetration}`
  - `touch_fill_fraction`
- Behavior:
  - `binary` preserves legacy touch/no-touch fills
  - `penetration` scales filled quantity by how far the bar traded through the limit
  - exact-touch bars can be forced to fill only partially or not at all via `touch_fill_fraction`
- Threaded the same knobs through:
  - `binanceexp1/run_multiasset_selector.py`
  - `binanceexp1/search_checkpoint_sets_robust.py`
  - `binanceexp1/sweep_multiasset_selector_robust.py`
  - `binanceexp1/train_multiasset_selector_robust.py`

### Tests
- Targeted validation:
  - `pytest tests/test_newnano_selector.py tests/test_selector_merged_simulation.py tests/test_binance_selector_robust_sweep.py tests/test_binance_train_multiasset_selector_robust.py tests/test_binance_checkpoint_set_search.py`
  - result: `34 passed`

### Real run started
- Started a real seeded robustness rerun for the current best baseline trio under:
  - `limit_fill_model=penetration`
  - `touch_fill_fraction=0.05`
  - same `lag=2`, `fill_buffer_bps=20`, `max_volume_fraction=0.1`, `realistic_selection=true`
- Output root:
  - `experiments/binance_selector_robust_penetration_baseline_20260308`
- At least one real fallback row was observed during the run:
  - `BTCUSD horizon=1h -> heuristic fallback 1/128 rows`

### Interpretation so far
- The simulator now has a more realistic way to model thin-liquidity passive fills than binary bar-touch logic.
- This should reduce optimistic fills especially on bars that only barely trade through the limit.
- The real robustness rerun is the next check for whether stricter fills materially worsen the current BTC/ETH/SOL selector baseline.
