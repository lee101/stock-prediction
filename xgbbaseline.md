# XGBoost Baseline

## Date
- 2026-03-29 UTC

## Goal
- Build an Alpaca-compatible hourly stock baseline that sits on top of Chronos2 forecasts.
- Optimize for smooth PnL and Sortino, not raw trade count.
- Use the current Alpaca production stack as the comparison bar, not the older hourly stock bot.

## Source of truth used
- `origin/main:alpacaprod.md`
- `origin/main:alpacaprogress6.md`
- `trade_daily_stock_prod.py`
- `unified_orchestrator/service_config.json`
- `unified_hourly_experiment/trade_unified_hourly.py`
- `unified_hourly_experiment/checkpoints/opt_retrain_v1/epoch_sweep_portfolio.json`
- `unified_hourly_experiment/checkpoints/retrain_stock_wd04_rw15/epoch_sweep_portfolio.json`
- `experiments/xgb_chronos_baseline_20260328_100634`
- `experiments/xgb_chronos_baseline_20260328_100737`
- `experiments/xgb_chronos_baseline_20260328_102535`
- `experiments/xgb_chronos_baseline_20260329_043939`
- `alpacafailedprogress6.md` was not present in this checkout and was not present on `origin/main`.

## Production baseline
- Live Alpaca stock production is the daily 6-model PPO ensemble, not the hourly Chronos2 stock bot.
- `origin/main:alpacaprod.md` documents the current canonical exhaustive marketsim for that live stock ensemble as:
- median `+58.0%`
- p10 `+45.4%`
- worst `+36.6%`
- negative windows `0 / 111`
- The hourly stock Chronos2 path is explicitly stopped in prod because recent validation remained negative on 7d+ windows.
- The active hourly stock symbol set owned by `trade-unified-hourly-meta` is:
- `DBX, TRIP, MTCH, NYT, NET, BKNG, EBAY, EXPE, ITUB, BTG, ABEV`

## Simulator and live audit
- The stale live `ETHUSD` incident is closed in broker history. `origin/main:alpacaprod.md` records the cancel at `2026-03-27 20:35:55 UTC` and the market sell fill at `2026-03-27 20:35:56 UTC`.
- The current hourly trading path already contains the main close-management fixes:
- crypto uses calendar-hour hold timing
- crypto zero-position detection uses notional, not `abs(qty) < 1`
- abandoned closes are retried through `pending_close_retry`
- cancel/replace paths wait before submitting the next order
- Verified coverage run before XGB work:
- `tests/test_trade_unified_hourly.py`
- `tests/test_replay_stock_trade_log_sim.py`
- `tests/test_portfolio_simulator_directional_amount.py`
- `tests/test_portfolio_simulator_goodness.py`
- `tests/test_portfolio_simulator_native_backend.py`
- `tests/test_trade_daily_stock_prod.py`
- Result: `78 passed`
- Verified again after the residual XGB changes:
- `tests/test_xgb_chronos_baseline.py`
- `tests/test_trade_unified_hourly.py`
- `tests/test_portfolio_simulator_directional_amount.py`
- `tests/test_trade_daily_stock_prod.py`
- Result: `63 passed`

## Legacy hourly stock baseline
- Stored checkpoint artifacts confirm the older hourly stock path is still the wrong baseline to deploy.
- `opt_retrain_v1` holdout portfolio results:
- `7d -1.73%`, Sortino `-9.01`
- `30d -5.16%`, Sortino `-8.73`
- `60d -7.48%`, Sortino `-3.67`
- `120d -15.50%`, Sortino `-12.55`
- `150d -19.02%`, Sortino `-9.05`
- `retrain_stock_wd04_rw15` epoch 1 was also negative across all short windows:
- `3d -1.65%`
- `7d -1.52%`
- `14d -1.83%`
- `30d -5.90%`
- Conclusion: the legacy hourly stock bot is materially worse than current prod and is not deployable.

## XGBoost experiments

### 1. First absolute XGB sweep
- Output: `experiments/xgb_chronos_baseline_20260328_100634`
- This version used Chronos2 features but predicted absolute targets rather than residual corrections.
- Best config:
- `label_horizon_hours=4`
- `risk_penalty=1.0`
- `entry_alpha=0.25`
- `exit_alpha=0.75`
- `edge_threshold=0.004`
- `max_hold_hours=2`
- `close_at_eod=true`
- `entry_selection_mode=first_trigger`
- `entry_allocator_mode=concentrated`
- Best metrics:
- `7d +0.1622%`, Sortino `4.2518`
- `14d -2.7555%`, Sortino `-14.5308`
- `30d -4.4874%`, Sortino `-5.2256`
- Result: too unstable and too lossy.

### 2. Conservative absolute XGB sweep
- Output: `experiments/xgb_chronos_baseline_20260328_100737`
- This was the best smoothness-oriented baseline before the residual refactor.
- Best config:
- `label_horizon_hours=4`
- `risk_penalty=1.25`
- `entry_alpha=0.0`
- `exit_alpha=1.0`
- `edge_threshold=0.008`
- `max_hold_hours=2`
- `close_at_eod=true`
- `entry_selection_mode=edge_rank`
- `entry_allocator_mode=concentrated`
- Best metrics:
- `7d +0.0000%`, Sortino `0.0000`, no trades
- `14d +0.0000%`, Sortino `0.0000`, no trades
- `30d -0.1609%`, Sortino `-0.4516`, max drawdown `1.6319%`
- trade count `9 buys / 9 sells`
- selection score `1.0268`
- Result: very conservative, very smooth, slightly negative over 30d, but still the best overall XGB baseline for the “smooth Sortino first” objective.

### 3. Residual XGB on top of blended Chronos2 priors
- Output: `experiments/xgb_chronos_baseline_20260328_102535`
- Changes made for this branch:
- blended Chronos2 priors from the available forecast horizons
- residual learning on top of those priors instead of replacing them
- richer forecast-shape features and horizon-gap features
- explicit search over `residual_scale` so `0.0` is a pure-Chronos control
- explicit search over `market_order_entry`

#### Best cross-window residual config
- `label_basis=reference_close`
- `label_horizon_hours=4`
- `residual_scale=0.5`
- `risk_penalty=1.5`
- `edge_threshold=0.010`
- `market_order_entry=true`
- `max_hold_hours=1`
- `close_at_eod=true`
- Metrics:
- `7d +1.0659%`, Sortino `9.3899`, max drawdown `1.2568%`
- `14d -2.1887%`, Sortino `-4.0628`, max drawdown `5.8125%`
- `30d -0.8371%`, Sortino `-0.8985`, max drawdown `3.8192%`
- trade count `97 buys / 97 sells`
- selection score `-4.8328`
- Result: it extracts short-window signal better than the older hourly models, but it is still worse than the conservative absolute XGB baseline on the 14d and 30d stability objective.

#### Pure Chronos control from the same sweep
- Best pure-Chronos control in that slice used `residual_scale=0.0`, `market_order_entry=true`, `max_hold_hours=2`, `edge_threshold=0.010`.
- Metrics:
- `7d +1.3764%`, Sortino `8.3180`
- `14d -7.5308%`, Sortino `-11.9870`
- `30d +1.7073%`, Sortino `2.6164`
- Result: this is the best 30d return in the focused residual sweep, but it is not smooth enough. The 14d loss is too large to accept as the baseline.

#### Full residual correction
- `residual_scale=1.0` performed materially worse than `0.5`.
- Best full-residual config still landed at:
- `30d -7.7979%`, Sortino `-8.2806`
- Result: full correction is overfitting or destabilizing the trade surface.

### 4. Probability-gated XGB with richer features
- Output: `experiments/xgb_chronos_baseline_20260329_043939`
- Changes made for this branch:
- added richer realized features: `2h/8h/48h` returns, short and long volatility, rolling range means, volume z-scores, and momentum-gap features
- added richer forecast features: upside/downside asymmetry, close-width, and close-skew features per Chronos horizon
- swapped the regressors to a more robust XGBoost objective (`pseudohuber`)
- trained XGB long/short direction classifiers and used them as a trade-probability gate
- searched `min_trade_probability` and `probability_power` to suppress low-conviction trades

#### Best probability-gated config
- `label_basis=reference_close`
- `label_horizon_hours=4`
- `residual_scale=0.0`
- `risk_penalty=1.5`
- `min_trade_probability=0.65`
- `probability_power=1.5`
- `edge_threshold=0.008`
- `market_order_entry=true`
- `max_hold_hours=1`
- `close_at_eod=true`
- Metrics:
- `7d +0.9559%`, Sortino `4.7354`, max drawdown `2.4435%`
- `14d -0.5312%`, Sortino `-1.1602`, max drawdown `3.1967%`
- `30d +2.9218%`, Sortino `5.1936`, max drawdown `3.2486%`
- trade count `45 buys / 44 sells`
- selection score `-0.0933`
- Result: this is the strongest hourly XGB research result so far.
- It improves materially on the previous residual branch:
- versus `20260328_102535`, `30d` improved from `-0.8371%` to `+2.9218%`
- versus `20260328_102535`, `14d` improved from `-2.1887%` to `-0.5312%`
- It also improves materially on the old pure-Chronos control:
- versus the old pure-Chronos control in `20260328_102535`, `30d` improved from `+1.7073%` to `+2.9218%`
- versus the old pure-Chronos control in `20260328_102535`, `14d` improved from `-7.5308%` to `-0.5312%`
- Important conclusion: the best current use of XGBoost is not price residual correction. It is probability gating and trade selection on top of Chronos prices.

## Current baseline decision
- Do not redeploy Alpaca stock production. The live daily PPO ensemble is still far stronger than anything in the hourly XGB path.
- Keep `experiments/xgb_chronos_baseline_20260328_100737` as the conservative guardrail because it is still the smoothest near-flat baseline.
- Promote `experiments/xgb_chronos_baseline_20260329_043939` to the leading research candidate because it is the best profitable hourly XGB result so far.
- Do not deploy `20260329_043939` yet because `14d` is still slightly negative and the live daily PPO stack remains much stronger.

## GPU pool prune baseline
- Added a hard-baseline early-prune path for RL sweeps in `pufferlib_market.gpu_pool` and `pufferlib_market.autoresearch_rl`.
- The pool now supports a stock-daily auto baseline profile:
- `baseline_val_return_floor = 0.35`
- `baseline_combined_floor = 1.0`
- `projection_clip_abs = 6.0`
- This baseline is intentionally below the established daily stock RL quick-eval winners:
- `stock_ent_03`: `val_return 0.5677`, `val_sortino 2.31`, combined `1.4389`
- `stock_ent_05`: `val_return 0.4807`, `val_sortino 2.30`, combined `1.3904`
- That makes it a beatable but still meaningful floor for remote GPU pruning.
- Important: this prune baseline is expressed in RL quick-validation metrics, not in the hourly XGB replay metrics above.
- Reason: the mid-training `run_trial()` probes only have fast C-env validation available at 25% / 50% / 75% budget. Copying the hourly XGB replay percentages directly into RL pruning would mix incompatible metric spaces and would prune for the wrong reason.
- The XGB hourly results still matter here. They are the research justification for demanding a stronger smoothness-and-selection standard, but the actual early-prune numbers must stay in the same scale as the mid-training validation signal.
- The projection path now also clips extreme combined-score spikes before fitting, so one noisy quick probe is less likely to keep a weak run alive.

## What this means
- Current production bar: very high and still not threatened by the hourly XGB path.
- Current hourly XGB bar for smoothness: nearly flat `30d -0.16%` with very low activity.
- Residual-on-Chronos price correction is no longer the leading idea.
- The better direction is Chronos price targets plus XGB feature engineering and XGB probability gating.
- The new best hourly candidate is now positive on `7d` and `30d`, with only a small `14d` loss left to solve.

## Next iterations
- Refine the probability-gated branch around `min_trade_probability` and `probability_power` to eliminate the remaining `14d` loss.
- Search per-symbol thresholds and hold times instead of one global threshold for all 11 Alpaca hourly stock symbols.
- Keep residual correction as a secondary branch, but treat it as optional. It should only survive if it beats the probability-gated pure-Chronos path.
- Test hybrid promotion rules such as “Chronos prices + XGB gate by default, residual correction only for symbols or regimes where the classifier shows persistent alpha.”
- Only consider deployment if a candidate is non-negative across `7d`, `14d`, and `30d` and improves Sortino versus both the `100737` guardrail and the new `20260329_043939` candidate.
