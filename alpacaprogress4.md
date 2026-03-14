# Alpaca Progress 4

## Goal

Test whether the hourly LLM trader improves when we stop over-constraining the model and when we calibrate forecast uncertainty using realized historical forecast error instead of the current Chronos quantile spread heuristic.

## New prompt variants

### `freeform`

- Keeps the same market context, constraints, and Chronos forecasts.
- Removes the strict/gated uncertainty rules.
- Tells the model to use the data however it thinks is best, as long as it clears fees and respects long-only + 6h max hold.

### `mae_bands`

- Uses the same freer prompt style as `freeform`.
- Replaces the dedicated uncertainty block with historical MAE-based error bands around the Chronos `p50` close forecast.
- The MAE band is causal in backtests:
  - It is computed from resolved historical forecasts only.
  - At decision timestamp `t`, only forecasts with `target_timestamp <= t` are used.
  - Default window is the last 30 days, falling back to all prior resolved forecasts when sample count is still small.

This mirrors the `../btcmarketsbot` homepage idea more closely than using the instantaneous `p10/p90` spread as a proxy for forecast reliability.

## Implementation notes

- Added `llm_hourly_trader/historical_error_bands.py`.
- Wired `mae_bands` into:
  - `llm_hourly_trader/experiment_runner.py`
  - `llm_hourly_trader/backtest.py`
- Added freer prompt variants to `llm_hourly_trader/gemini_wrapper.py`:
  - `freeform`
  - `mae_bands`
- Made `gemini_wrapper.py` import-safe for prompt-only/test contexts when `google-genai` is not installed.

## Verification

- `pytest -q tests/test_llm_hourly_trader_historical_error_bands.py`
- `pytest -q tests/test_llm_forecast_pipeline.py`

Both passed after making `gemini_wrapper.py` resilient to missing SDK imports during prompt-only tests.

## Pilot run

Sequential 1-day pilot on `BTCUSD`, comparing:

- `uncertainty_strict`
- `freeform`
- `mae_bands`

Command:

```bash
source .venv/bin/activate
python -m llm_hourly_trader.experiment_runner \
  --symbols BTCUSD \
  --days 1 \
  --variants uncertainty_strict freeform mae_bands \
  --model gemini-3.1-flash-lite-preview \
  --rate-limit 4.2
```

Status:

- Attempted.
- Did not complete in this shell session because the provider call stayed in a long-lived HTTPS session upstream and never returned a comparison summary in reasonable wall clock time.
- No backtest metrics recorded yet from this pilot.

## Real-data smoke check

Built a `mae_bands` prompt on current `BTCUSD` cache data without hitting the LLM provider:

- Decision timestamp: `2026-03-13 22:00:00+00:00`
- 1h historical MAE band: `0.4167%` from `721` resolved samples
- 24h historical MAE band: `2.8245%` from `721` resolved samples
- Prompt length: `3341` chars

This confirms the causal MAE-band path works on real forecast caches and current hourly history.

## Fixed-window BTCUSD comparison

To make the comparison reproducible, added fixed `--end-ts` support and cache-only replay support to the hourly trader runners.

Provider-backed run:

```bash
source .venv/bin/activate
python -m llm_hourly_trader.experiment_runner \
  --symbols BTCUSD \
  --days 1 \
  --variants uncertainty_strict freeform mae_bands \
  --model gemini-3.1-flash-lite-preview \
  --rate-limit 0 \
  --end-ts 2026-03-14T00:00:00Z
```

Window:

- `2026-03-13 00:00:00+00:00 -> 2026-03-14 00:00:00+00:00`

Result:

- `uncertainty_strict`: `+0.000%`, `0` trades
- `freeform`: `+0.183%`, `6` trades, `Sortino 128.50`, realized PnL `+$4.49` on `$2,000`
- `mae_bands`: `+0.098%`, `2` trades, `Sortino 16.91`, realized PnL `+$2.22` on `$2,000`

Interpretation:

- On this fixed BTC window, the freer prompt is better than the strict uncertainty gate and better than the MAE-band framing.
- The MAE-band prompt still trades and stays profitable here, but it is more selective and captures less upside than `freeform`.

Cache-only replay:

```bash
source .venv/bin/activate
python -m llm_hourly_trader.experiment_runner \
  --symbols BTCUSD \
  --days 1 \
  --variants uncertainty_strict freeform mae_bands \
  --model gemini-3.1-flash-lite-preview \
  --rate-limit 0 \
  --end-ts 2026-03-14T00:00:00Z \
  --cache-only
```

The cache-only replay reproduced the exact same metrics, confirming the fixed-window replay path works for prompt iteration without another live provider pass.


## Earlier Progress

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

## 2026-03-11 16:34 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_nonyt | edge_rank | 2.2255 | 2.9859 | 0.2345 | 0.4892 | 18 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_nonyt | edge_rank | 2.2255 | 2.9859 | 0.2345 | 0.4892 | 18 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_nonyt | edge_rank | 1.5652 | 2.8210 | 0.1726 | 0.4887 | 20 | 0.0 | 0.0 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_nonyt | edge_rank | 1.1746 | 2.3934 | 0.1298 | 0.4459 | 20 | 0.0 | 0.0 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7 | first_trigger | -1.7884 | 0.1064 | -0.4410 | 0.0411 | 18 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-11 17:53 UTC Autonomous Research Batch (4 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_nonyt | first_trigger | 1.5543 | 2.8133 | 0.1726 | 0.4887 | 20 | 0.0 | 0.0 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_nonyt | first_trigger | 1.1746 | 2.3987 | 0.1298 | 0.4459 | 20 | 0.0 | 0.0 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_nonyt | first_trigger | -3.1815 | -1.8454 | -1.3554 | -0.8733 | 18 | 0.0 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_nonyt | first_trigger | -3.1815 | -1.8454 | -1.3554 | -0.8733 | 18 | 0.0 | -0.002 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-11 21:11 UTC Autonomous Research Batch (8 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7 | edge_rank | 2.2255 | 3.8169 | 0.2345 | 0.6783 | 16 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7 | edge_rank | 2.2255 | 3.8169 | 0.2345 | 0.6783 | 16 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7 | edge_rank | 1.5543 | 2.9721 | 0.1726 | 0.5034 | 18 | 0.0 | 0.0 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7 | edge_rank | 1.2644 | 3.4820 | 0.1438 | 0.6959 | 18 | 0.0025 | -0.002 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7 | edge_rank | -2.3343 | -1.2390 | -0.9032 | -0.4902 | 16 | 0.005 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 00:22 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_nonyt | edge_rank | 2.2255 | 3.8169 | 0.2345 | 0.6783 | 16 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_nonyt | edge_rank | 2.2255 | 3.8169 | 0.2345 | 0.6783 | 16 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7 | first_trigger | 1.5543 | 2.9721 | 0.1726 | 0.5034 | 18 | 0.0 | 0.0 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_nonyt | edge_rank | 1.5543 | 2.9850 | 0.1726 | 0.5170 | 20 | 0.0 | 0.0 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7 | first_trigger | 1.2644 | 3.4820 | 0.1438 | 0.6959 | 18 | 0.0025 | -0.002 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 03:28 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_nonyt | first_trigger | 1.5543 | 2.9850 | 0.1726 | 0.5170 | 20 | 0.0 | 0.0 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7 | edge_rank | 1.5543 | 2.9676 | 0.1726 | 0.5034 | 18 | 0.0 | 0.0 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_nonyt | first_trigger | 1.2644 | 3.4051 | 0.1438 | 0.6901 | 16 | 0.005 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7 | edge_rank | 1.2644 | 3.4743 | 0.1438 | 0.6959 | 18 | 0.0025 | -0.002 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_nonyt | first_trigger | -1.5170 | -1.0870 | -0.7589 | -0.5830 | 18 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 06:29 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7 | edge_rank | 2.1305 | 2.6781 | 0.2789 | 0.2789 | 20 | 0.0 | 0.0 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7 | edge_rank | 2.1305 | 2.6781 | 0.2789 | 0.2789 | 20 | 0.0 | 0.0 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7 | first_trigger | 1.5543 | 2.9676 | 0.1726 | 0.5034 | 18 | 0.0 | 0.0 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7_nonyt | edge_rank | 1.5543 | 2.9803 | 0.1726 | 0.5170 | 20 | 0.0 | 0.0 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7 | first_trigger | 1.2644 | 3.4743 | 0.1438 | 0.6959 | 18 | 0.0025 | -0.002 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 09:56 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7_nonyt | edge_rank | 2.1305 | 2.6781 | 0.2789 | 0.2789 | 20 | 0.0 | 0.0 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7_nonyt | edge_rank | 2.1305 | 2.6781 | 0.2789 | 0.2789 | 20 | 0.0 | 0.0 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7_nonyt | first_trigger | 1.5543 | 2.9803 | 0.1726 | 0.5170 | 20 | 0.0 | 0.0 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7_nonyt | first_trigger | 1.2644 | 3.4743 | 0.1438 | 0.6959 | 24 | 0.0025 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,MSFT | prod7_nonyt | edge_rank | -1.5170 | -0.5265 | -0.7589 | -0.3108 | 18 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 13:08 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7 | edge_rank | 2.2255 | 3.0912 | 0.2345 | 0.4892 | 18 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7 | edge_rank | 2.2255 | 3.0912 | 0.2345 | 0.4892 | 18 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7 | edge_rank | -3.0748 | -1.7621 | -1.1397 | -0.7279 | 18 | 0.0 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7 | edge_rank | -3.0748 | -1.7621 | -1.1397 | -0.7279 | 18 | 0.0 | -0.002 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7 | edge_rank | -6.3223 | -3.5582 | -3.5761 | -2.2800 | 18 | 0.0 | 0.0 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 16:17 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | edge_rank | 2.1471 | 2.6981 | 0.2789 | 0.2789 | 18 | 0.0 | 0.0 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | edge_rank | 2.1471 | 2.6981 | 0.2789 | 0.2789 | 18 | 0.0 | 0.0 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | edge_rank | -4.3268 | -3.5483 | -1.7008 | -1.4612 | 14 | 0.0 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | edge_rank | -4.3268 | -3.7314 | -1.8204 | -1.5409 | 16 | 0.0 | -0.002 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | edge_rank | -6.4877 | -3.3192 | -3.7499 | -2.1688 | 22 | 0.0025 | 0.0 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 17:32 UTC Autonomous Research Batch (4 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | first_trigger | -6.5096 | -3.3280 | -3.7499 | -2.1688 | 22 | 0.0025 | 0.0 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | first_trigger | -6.5107 | -3.3287 | -3.7499 | -2.1688 | 22 | 0.0025 | 0.0 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | first_trigger | -8.1555 | -5.4972 | -4.6145 | -3.1485 | 18 | 0.0 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NET | prod7_nonyt | first_trigger | -8.1555 | -5.4972 | -4.6145 | -3.1485 | 18 | 0.0 | -0.002 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 18:47 UTC Autonomous Research Batch (2 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7 | edge_rank | 1.3980 | 2.8204 | 0.1781 | 0.5440 | 16 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7 | edge_rank | 1.3980 | 2.8204 | 0.1781 | 0.5440 | 16 | 0.005 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-12 PufferLib RL Causal Fix + OOS Autoresearch

### Critical Finding: Look-Ahead Bias
The PufferLib C trading environment had a look-ahead bias where the agent saw the current bar's OHLC then traded at the same bar's prices. Fixed with:
1. **1-bar observation lag**: agent sees `t-1` features, trades at `t`
2. **OPEN price execution**: orders reference bar's open (not close)
3. **Fill slippage**: added configurable adverse slippage (buys fill higher, sells lower)

Previous "record" returns (2,659x/30d) were inflated. Retrained models still strong in-sample but **overfit**: crypto4 model scored +89x in-sample vs -14.9%/30d on unseen data (0% profitable).

### OOS Autoresearch (5-min timeboxed training, eval with 8bps slippage on held-out Jun 2025-Feb 2026)

**Data**: 5 crypto symbols (BTC, ETH, SOL, LTC, AVAX), Train: 2022-2025 (29,929h), Val: Jun 2025-Feb 2026 (6,001h)

**31 trials completed. Full OOS ranking (positive only):**

| Rank | Config | Val Return/30d | Val Sortino | Val Profitable% | Key Change |
|------|--------|---------------|-------------|-----------------|------------|
| 1 | **slip_5bps** | **+5.30%** | **1.62** | **96%** | 5bps fill slippage |
| 2 | **ent_01** | **+4.87%** | **1.56** | **93%** | Higher entropy (0.1 vs 0.05) |
| 3 | reg_combo_2 | +3.69% | 1.47 | 91% | obs_norm + wd=0.05 + slip=8bps |
| 4 | obs_norm | +3.03% | 1.68 | 78% | Observation normalization |
| 5 | lr_1e4 | +2.31% | 1.23 | 71% | Lower LR (1e-4 vs 3e-4) |
| 6 | reg_combo_1 | +2.22% | 0.96 | 82% | wd=0.01 + slip=8bps + trade_pen |
| 7 | clip_vloss | +1.77% | 0.88 | 70% | Clip value loss |
| 8 | reg_combo_3 | +0.95% | 1.19 | 67% | obs_norm + ent=0.08 + cosine |
| 9 | kitchen_sink | +0.95% | 1.19 | 67% | Everything combined |
| 10 | smooth_ds | +0.42% | 0.61 | 57% | Smooth downside penalty |

**Negative configs (overfit or collapsed):** baseline (-0.83%), cosine_lr (-1.23%), seed variants (-1.63%, -12.4%), ent_anneal (-14.75%), gamma_999 (-6.3%), h256 (-4.55%), trade_pen_05 (-7.25%), all wd configs without obs_norm

**Key insights**:
1. Training with execution friction (slippage, higher fees) forces wider edges that generalize
2. Higher entropy (0.1) prevents premature policy collapse, close to #1
3. obs_norm is the strongest single regularizer
4. Lower LR (1e-4) also helps generalization
5. Seeds matter: different seeds on same config produce -12.4% to -0.83%
6. Baseline anneal-LR alone is insufficient for OOS (-0.83%)
7. Combining multiple regularizers (reg_combo_2) is strong but not better than focused interventions

### RL+Gemini Hybrid Backtest (Gemini 3.1 Flash Lite, Thinking=HIGH)
- rl_gemini: -2.40%, Sortino -7.62, 56 fills, 256 API calls
- gemini_only: -3.56%, Sortino -9.28, 42 fills, 507 API calls
- RL+Gemini wins: better returns, half the API calls

### Momentum Proxy Backtest on HourlyTrader Simulator (Mar 4-11 2026)
- **rl_only** (SMA crossover proxy, 0.2%/0.8% spread): -7.38%, Sortino=-14.00, WR=58.1%, 31 trades, MaxDD=8.48%
- Period was bearish (BTC -1.6%, ETH/SOL down more)
- Gemini API was unresponsive during this session — rl_gemini and gemini_only comparison pending
- Created `unified_orchestrator/backtest_rl_gemini_real.py` for proper 3-mode comparison

### Currently Running (as of 22:00 UTC)
- **Autoresearch loop**: 25/35 trials, currently training `ent_01` (high entropy 0.1)
- **crypto6_obsnorm_200M**: Full training, ~101M/200M steps (ret=+951x in-sample, step 101.5M)
- **crypto6_slip5_200M**: Full training with slippage, ~49M/200M steps (ret=+5.97x in-sample)
- **Stock trader (Alpaca)**: Live, meta-selector with 7 strategies
- **Crypto trader (Binance)**: Live
- **Unified orchestrator**: Live (--live mode)

### Current Production Algorithms
- **Stocks**: Neural policy meta-selector (wd_0.06_s42/epoch_008.pt), 6 symbols, hold=5h, edge=0.008
- **Crypto**: Unified orchestrator with RL+Gemini hybrid (momentum proxy, awaiting OOS-validated RL model)


## 2026-03-12 22:01 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7 | first_trigger | 1.3980 | 2.8328 | 0.1781 | 0.5440 | 16 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7 | first_trigger | 1.3980 | 2.8328 | 0.1781 | 0.5440 | 16 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7 | edge_rank | -2.6664 | -1.8392 | -1.1073 | -0.7815 | 18 | 0.0 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7 | edge_rank | -2.6664 | -1.8392 | -1.1073 | -0.7815 | 18 | 0.0 | -0.002 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7 | first_trigger | -4.2755 | -2.1599 | -1.8020 | -1.0994 | 14 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.

## 2026-03-13 200M Step OOS Evaluation — Overfitting Confirmed

### Background
Trained two 200M step models on crypto6 train data (2022-2025, 29,929h) to test whether longer training improves OOS generalization:
- **crypto6_obsnorm_200M**: obs_norm=True, no slippage, 200M steps → 7,340x in-sample return
- **crypto6_slip5_200M**: fill_slippage=5bps, 200M steps → 347x in-sample return

### OOS Results (Jun 2025 - Feb 2026, 6,001h, 8bps eval slippage)

| Model | OOS Return/30d | Profitable% | In-Sample Return | Verdict |
|-------|---------------|-------------|-----------------|---------|
| **autoresearch/slip_5bps (5min)** | **+5.21% median** | **96%** | modest | **BEST — generalizes** |
| crypto6_obsnorm_200M (no slip) | -3.24% mean | 9% | 7,340x | Overfit |
| crypto6_obsnorm_200M (8bps slip) | -1.71% median | ~25% | 7,340x | Overfit |
| crypto6_slip5_200M (8bps slip) | -16.1% mean | 0% | 347x | Severely overfit |

### Independent confirmation of autoresearch best (slip_5bps with 8bps eval slippage)
- Median return: +5.21%/30d
- p25: +2.44%, p75: +8.40%
- Best episode: +14.33%, Worst: -4.86%
- 96% profitable episodes

### Key Insight: More Training Steps = More Overfitting
- 200M step models memorize training data patterns that don't generalize
- 5-minute timeboxed training (~8-9M steps) produces far better OOS performance
- Training with execution friction (slippage) is necessary but NOT sufficient — must also limit training duration
- The autoresearch approach (short training + diverse configs + OOS evaluation) is the correct methodology

### Currently Running (as of 2026-03-13 00:30 UTC)
- **Stock trader (Alpaca)**: Live, meta-selector with 7 strategies (PID 3882689)
- **Unified orchestrator**: Live, CRYPTO_ONLY regime (PID 2011069)
- **Data collector**: 5min bars (PID 2131798)
- **Autoresearch**: Complete (35/35 trials)
- **200M training**: Complete (both runs finished, OOS evaluated)

### Current Production Algorithms
- **Stocks**: Neural policy meta-selector (wd_0.06_s42/epoch_008.pt), 6 symbols, hold=5h, edge=0.008
- **Crypto**: Unified orchestrator — awaiting deployment of OOS-validated RL model (slip_5bps)

### Deployment: slip_5bps RL Model → Production (2026-03-13 00:36 UTC)

Deployed OOS-validated RL model as signal hint to crypto orchestrator:
- **Model**: `pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt`
- **OOS performance**: +5.21% median/30d, 96% profitable, Sortino 1.62
- **Architecture**: TradingPolicy MLP h1024, 5 symbols (BTC/ETH/SOL/LTC/AVAX), 11 actions
- **Integration**: RL generates direction+confidence signals, injected as hints into Gemini Flash prompt
- **Code changes**:
  - `unified_orchestrator/orchestrator.py`: Added `_get_crypto_rl_trader()`, updated `CRYPTO_CHECKPOINT_CANDIDATES` and `CRYPTO_SYMBOLS`
  - `pufferlib_market/inference.py`: Added auto-detection of TradingPolicy vs ResidualPolicy architecture
- **First live signals**: BTC long (0.68), ETH short (0.30), SOL long (1.00), LTC long (0.31), AVAX long (0.19)
- **Orchestrator PID**: 2849387

### Next Steps
1. Monitor RL+LLM hybrid performance over next 24-48h
2. Re-run RL+Gemini backtest when Gemini API is stable
3. Consider training combined best techniques (slip + ent_01 + obs_norm) with SHORT timeboxing
4. Backtest RL+Gemini on stocks (pending)
5. Fix position sizing for 5 symbols (SOL order failed due to 3-way equal split exceeding cash)


## 2026-03-13 02:08 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7_nonyt | edge_rank | 1.3980 | 2.8328 | 0.1781 | 0.5440 | 16 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7_nonyt | edge_rank | 1.3980 | 2.8328 | 0.1781 | 0.5440 | 16 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7_nonyt | first_trigger | 1.3980 | 2.8328 | 0.1781 | 0.5440 | 16 | 0.005 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7_nonyt | first_trigger | 1.3980 | 2.8328 | 0.1781 | 0.5440 | 16 | 0.005 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7_nonyt | edge_rank | -1.5624 | -0.7016 | -0.7586 | -0.4046 | 18 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-13 04:50 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7 | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7 | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.002 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7 | edge_rank | 0.8017 | 1.1228 | 0.1945 | 0.2262 | 14 | 0.005 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7 | edge_rank | -0.8437 | 0.5553 | -0.1394 | 0.0643 | 16 | 0.005 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,TSLA | prod7_nonyt | first_trigger | -1.5624 | -0.7016 | -0.7586 | -0.4046 | 18 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-13 07:37 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7 | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7 | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.002 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7_nonyt | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.005 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7_nonyt | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 24 | 0.0025 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7_nonyt | edge_rank | 0.8017 | 1.1228 | 0.1945 | 0.2262 | 14 | 0.005 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-13 10:28 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7_nonyt | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7_nonyt | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 24 | 0.0025 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7 | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7 | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.002 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH | prod7_nonyt | first_trigger | -1.5306 | -0.7367 | -0.7589 | -0.4306 | 18 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-13 13:24 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7 | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7 | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.002 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7_nonyt | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.005 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7_nonyt | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 24 | 0.0025 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7 | edge_rank | 0.8017 | 1.1228 | 0.1945 | 0.2262 | 14 | 0.005 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-13 16:29 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7_nonyt | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7_nonyt | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 24 | 0.0025 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7_nonyt | edge_rank | 0.8016 | 1.1226 | 0.1945 | 0.2262 | 14 | 0.005 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7_nonyt | edge_rank | -0.8453 | 0.5544 | -0.1394 | 0.0643 | 16 | 0.005 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT | prod7_nonyt | edge_rank | -1.5306 | -0.7367 | -0.7589 | -0.4306 | 18 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-13 18:57 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | core5 | edge_rank | 2.9381 | 3.3502 | 0.2976 | 0.5012 | 18 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | core5 | edge_rank | 2.9381 | 3.3502 | 0.2976 | 0.5012 | 18 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | core5 | first_trigger | 1.0578 | 1.2604 | 0.1912 | 0.1912 | 14 | 0.005 | 0.0 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | core5 | edge_rank | 1.0530 | 1.2588 | 0.1912 | 0.1912 | 14 | 0.005 | 0.0 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | core5 | first_trigger | 0.7198 | 1.0220 | 0.1022 | 0.1342 | 20 | 0.0075 | -0.001 |

### NEW RECORD - Beats baseline!
- symbols: `NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL`
- strategies: `core5`
- entry_mode: `edge_rank`
- min_goodness: **2.9381** (was 2.7965)
- mean_goodness: **3.3502**
- min_return: **0.2976%**
- mean_return: **0.5012%**
- Best params: lookback=18, switch=0.005, sit_out=-0.001, mode=sticky
- Supervisor conf updated.


## 2026-03-13 22:26 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_all | edge_rank | 1.1880 | 2.1386 | 0.1484 | 0.4214 | 22 | 0.0075 | -0.002 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_all | edge_rank | 1.0602 | 1.2612 | 0.1912 | 0.1912 | 14 | 0.005 | 0.0 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | core5 | first_trigger | 0.3862 | 0.5305 | 0.0477 | 0.0613 | 18 | 0.0 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | core5 | first_trigger | 0.3862 | 0.5305 | 0.0477 | 0.0613 | 18 | 0.0 | -0.002 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_all | edge_rank | -0.8491 | -0.5373 | -0.1394 | -0.1394 | 18 | 0.0025 | 0.0 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-14 01:56 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_all | first_trigger | 1.1880 | 2.1436 | 0.1484 | 0.4214 | 22 | 0.0075 | -0.002 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_all | first_trigger | 1.0651 | 1.2628 | 0.1912 | 0.1912 | 14 | 0.005 | 0.0 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_wd07 | edge_rank | 1.0651 | 1.2628 | 0.1912 | 0.1912 | 14 | 0.005 | 0.0 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_wd07 | edge_rank | 0.7237 | 1.0233 | 0.1022 | 0.1342 | 20 | 0.0075 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_wd07 | edge_rank | 0.4239 | 0.7264 | 0.0564 | 0.0956 | 20 | 0.005 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-14 03:51 UTC Autonomous Research Batch (6 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_wd07 | first_trigger | 1.0651 | 1.2628 | 0.1912 | 0.1912 | 14 | 0.005 | 0.0 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_wd07 | first_trigger | 0.7237 | 1.0233 | 0.1022 | 0.1342 | 20 | 0.0075 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_wd07 | first_trigger | -1.1933 | -0.3551 | -0.5092 | -0.2950 | 14 | 0.0 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_wd07 | edge_rank | -2.5416 | -1.8724 | -1.1522 | -0.8468 | 14 | 0.0 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL | prod7_wd07 | edge_rank | -4.1037 | -3.3784 | -1.7348 | -1.4023 | 18 | 0.0 | -0.002 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-14 05:21 UTC Autonomous Research Batch (2 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | core5 | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | core5 | edge_rank | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.002 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-14 07:56 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | core5 | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | core5 | first_trigger | 1.3980 | 2.7138 | 0.1781 | 0.4917 | 18 | 0.0025 | -0.002 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | core5 | edge_rank | -0.7324 | 1.1640 | -0.2523 | 0.0869 | 16 | 0.005 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | core5 | edge_rank | -0.8496 | -0.5487 | -0.1394 | -0.1394 | 18 | 0.0025 | 0.0 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | core5 | edge_rank | -2.5139 | -1.7963 | -1.1130 | -0.7911 | 14 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-14 11:55 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_all | edge_rank | 1.3980 | 2.6627 | 0.1781 | 0.4860 | 16 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_all | edge_rank | 1.3980 | 2.6627 | 0.1781 | 0.4860 | 16 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_all | first_trigger | 1.3980 | 2.6627 | 0.1781 | 0.4860 | 16 | 0.005 | -0.001 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_all | first_trigger | 1.3980 | 2.6627 | 0.1781 | 0.4860 | 16 | 0.005 | -0.001 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_all | edge_rank | -0.8496 | -0.5487 | -0.1394 | -0.1394 | 18 | 0.0025 | 0.0 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.


## 2026-03-14 15:35 UTC Autonomous Research Batch (10 experiments)

Baseline: min_goodness=2.7965, mean_goodness=3.9151, min_return=0.4139%


### Top-5 results this batch

| Rank | Symbols | Strategies | Entry | min_goodness | mean_goodness | min_ret% | mean_ret% | lookback | switch | sit_out |
|------|---------|-----------|-------|-------------|--------------|---------|---------|---------|--------|--------|
| 1 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_wd07 | edge_rank | 1.3980 | 2.6627 | 0.1781 | 0.4860 | 16 | 0.005 | -0.001 |
| 2 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_wd07 | edge_rank | 1.3980 | 2.6627 | 0.1781 | 0.4860 | 16 | 0.005 | -0.001 |
| 3 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_wd07 | edge_rank | -0.8496 | -0.5487 | -0.1394 | -0.1394 | 18 | 0.0025 | 0.0 |
| 4 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_wd07 | edge_rank | -0.8496 | -0.5487 | -0.1394 | -0.1394 | 18 | 0.0025 | 0.0 |
| 5 | NVDA,PLTR,GOOG,DBX,TRIP,MTCH,META | prod7_wd07 | edge_rank | -1.9789 | -0.6887 | -0.8529 | -0.4096 | 14 | 0.0 | -0.001 |

### Deploy decision
- No config exceeded baseline min_goodness=2.7965 - continuing search.

