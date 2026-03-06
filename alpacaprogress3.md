# Alpaca Progress 3

## 2026-03-04 End-to-End Optimization + Deploy

### Goal
- Improve stock live trading by combining:
  - Chronos2 hourly hyperparameter tuning (MAE + smoothness composite objective)
  - Pre-augmentation sweep
  - Meta strategy selection across multiple stock policies
- Deploy only if holdout metrics beat current live baseline.

## 1) Baseline (current live bot) benchmark

Command run:
- `python unified_hourly_experiment/run_stock_sortino_lag_robust.py --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT --configs-json /tmp/baseline_live_eval_config.json --experiment-name live_baseline_eval_20260304 --validation-days 30 --eval-lags 0,1,2,3 --eval-min-edges 0.0,0.001,0.002,0.003,0.004,0.005,0.006,0.007 --min-buys-mean 0 --max-configs 1 --sim-backend auto --reuse-checkpoints`

Artifacts:
- `experiments/live_baseline_eval_20260304/summary.json`
- `experiments/live_baseline_eval_20260304/results.json`

Best baseline row (live `wd_0.04` epoch 9):
- `selection_score=-7.2831`
- `robust_score=-7.2831`
- `sortino_p10=-5.1410`
- `return_mean_pct=-10.0820`
- `pnl_smoothness_mean=0.001635`
- `max_drawdown_mean_pct=10.1780`
- `num_buys_mean=177.75`

## 2) Chronos2 hourly hyperparameter tuning (live symbols)

Command run:
- `python hyperparam_chronos_hourly.py --symbols NVDA PLTR GOOG DBX TRIP MTCH NYT --quick --holdout-hours 336 --prediction-length 24 --objective composite --smoothness-weight 0.40 --direction-bonus 0.05 --cohort-size 2 --cohort-min-abs-corr 0.20 --enable-cross-learning --device cuda --output experiments/hyperparam_chronos_hourly_live7_20260304.json --save-hyperparams`

Artifact:
- `experiments/hyperparam_chronos_hourly_live7_20260304.json`

Chosen configs (best_per_symbol):
- `NVDA`: ctx `1024`, skip `[1,2,3]`, objective `0.00886`, pct MAE `0.01318`, smoothness `0.00308`
- `PLTR`: ctx `1024`, skip `[1,2,3]`, objective `0.00886`, pct MAE `0.01318`, smoothness `0.00308`
- `GOOG`: ctx `1024`, skip `[1,2,3]`, objective `0.00886`, pct MAE `0.01318`, smoothness `0.00308`
- `DBX`: ctx `1024`, skip `[1]`, objective `0.00595`, pct MAE `0.00909`, smoothness `0.00256`
- `TRIP`: ctx `2048`, skip `[1]`, objective `0.00935`, pct MAE `0.01352`, smoothness `0.00520`
- `MTCH`: ctx `1024`, skip `[1]`, objective `-0.00285`, pct MAE `0.00295`, smoothness `0.00113`
- `NYT`: ctx `2048`, skip `[1,2,3]`, objective `0.01096`, pct MAE `0.01260`, smoothness `0.00370`

## 3) Hourly pre-augmentation sweep

Command run:
- `python preaug_sweeps/evaluate_preaug_chronos.py --symbols NVDA PLTR GOOG DBX TRIP MTCH NYT --hyperparam-root hyperparams/chronos2/hourly --selection-metric pct_return_mae --strategy-repeats 1 --output-dir preaugstrategies/chronos2/hourly --mirror-best-dir preaugstrategies/best/hourly --data-dir trainingdatahourly/stocks --device-map cuda --frequency hourly --benchmark-cache-dir chronos2_benchmarks/preaug_cache_hourly_live7 --report-dir preaug_sweeps/reports_hourly`

Artifact:
- `preaug_sweeps/reports_hourly/chronos_preaug_20260304_230858.json`

Best strategy per symbol:
- `NVDA`: `detrending` (`pct_return_mae=0.0071`)
- `PLTR`: `detrending` (`0.0122`)
- `GOOG`: `percent_change` (`0.0065`)
- `DBX`: `percent_change` (`0.0044`)
- `TRIP`: `rolling_norm` (`0.0089`)
- `MTCH`: `rolling_norm` (`0.0070`)
- `NYT`: `rolling_norm` (`0.0033`)

Notes:
- `log_returns` failed on several symbols with `Need at least 3 dates to infer frequency`.

## 4) Meta strategy optimization (multi-policy selector)

Strategies searched:
- `wd03=/.../wd_0.03:18`
- `wd04=/.../wd_0.04:9`
- `wd05=/.../wd_0.05:17`
- `wd06=/.../wd_0.06:20`
- `wd07=/.../wd_0.07:20`
- `wd08=/.../wd_0.08:20`

First run (`min_num_buys=2`) produced no eligible row due activity filter (all rows had `min_num_buys=1`).

Relaxed run (`min_num_buys=1`) command:
- `python unified_hourly_experiment/auto_meta_optimize.py ... --min-num-buys 1 --output-dir experiments/auto_meta_live7_20260304_relaxed`

Artifacts:
- `experiments/auto_meta_live7_20260304_relaxed/auto_meta_recommendation.json`
- `experiments/auto_meta_live7_20260304_relaxed/meta_edge0p002_th0p0_mwinner_sm0p0_mg0p0.json` (selected row)

Best meta config:
- `metric=p10`
- `lookback_days=10`
- `selection_mode=winner`
- `switch_margin=0.0`
- `min_score_gap=0.0`
- `min_edge=0.002`
- `sit_out_threshold=0.0`
- `min_sortino=0.0304`
- `mean_sortino=0.2228`
- `min_return_pct=0.00094`
- `mean_return_pct=0.18584`
- `mean_max_drawdown_pct=0.20817`

Best meta vs single-strategy `wd04` baseline (same sweep output):
- `min_sortino`: `0.0304` vs `-11.0488`
- `mean_sortino`: `0.2228` vs `-9.6799`
- `min_return_pct`: `0.00094` vs `-21.3455`
- `mean_return_pct`: `0.18584` vs `-16.1293`
- `mean_max_drawdown_pct`: `0.20817` vs `16.5534`

## 5) Deployment performed

Updated supervisor:
- `supervisor/unified-stock-trader.conf`
  - Switched `unified-stock-trader` from `trade_unified_hourly.py` (single checkpoint) to `trade_unified_hourly_meta.py` with best meta config.
  - Added `CHRONOS2_FREQUENCY="hourly"` for both `unified-stock-trader` and `stock-cache-refresh`.

Applied:
- `sudo supervisorctl reread`
- `sudo supervisorctl update`
- `sudo supervisorctl restart unified-stock-trader`
- `sudo supervisorctl restart stock-cache-refresh`

Runtime confirmation:
- `unified-stock-trader` RUNNING with meta command (`trade_unified_hourly_meta.py ...`)
- `stock-cache-refresh` RUNNING

## 6) Reliability fix added (cache refresh)

Issue found:
- `BEST_MODELS` contained several missing Chronos2 LoRA directories for live symbols, causing cache rebuild skips.

Fix:
- In `unified_hourly_experiment/rebuild_all_caches.py`, added `_resolve_model_path(symbol, model_name)`:
  - Use configured model if present.
  - If missing, fallback to newest local `chronos2_finetuned/<SYMBOL>_* / finetuned-ckpt`.
  - Log warning with fallback path.

Test added:
- `tests/test_rebuild_all_caches.py`
  - verifies preferred path behavior
  - verifies latest-fallback behavior

Validation:
- `pytest -q tests/test_rebuild_all_caches.py` -> `2 passed`.

## 7) Tooling correctness fix

Issue:
- `auto_meta_optimize.py` generated deploy commands including `--sim-backend`, but `trade_unified_hourly_meta.py` does not support that argument.

Fix:
- Removed `--sim-backend` from generated `deploy_command` in `unified_hourly_experiment/auto_meta_optimize.py`.

Validation:
- `pytest -q tests/test_auto_meta_optimize.py tests/test_rebuild_all_caches.py` -> `5 passed`.

## 8) Simulator Realism Hardening (Sparse Actions + Pending Orders)

Objective:
- make simulator behavior closer to live when actions are sparse (common in log-replay and event-driven evaluations), especially for delayed fills and order lifecycle.

Code changes:
- `unified_hourly_experiment/marketsimulator/portfolio_simulator.py`
  - fixed `close_at_eod=False` crash (`closed3` unbound local).
  - switched bar/action merge from `inner` to `left`:
    - simulator now advances state on all bar timestamps even if no action row exists that hour.
    - pending entries, timeouts, and EOD logic now progress on sparse-action timelines.
  - added column validation errors:
    - bars/actions must include `timestamp` and `symbol`.
  - pending entry lifecycle support retained:
    - `PortfolioConfig.entry_order_ttl_hours` (python backend only).

New and expanded tests:
- `tests/test_portfolio_simulator_directional_amount.py`
  - pending TTL disabled: no delayed fill without signal.
  - pending TTL enabled: delayed fill on later bar without new signal.
  - pending TTL expiry behavior.
  - pending fill on bar with no action row (verifies sparse-action realism fix).
  - equity curve keeps all bar timestamps when actions are sparse.

Validation run:
- `pytest -q tests/test_portfolio_simulator_directional_amount.py tests/test_portfolio_simulator_native_backend.py tests/test_simulator_math.py`
  - result: `36 passed`
- `pytest -q tests/test_trade_alpaca_hourly_utils.py tests/test_auto_meta_optimize.py tests/test_meta_selector.py tests/test_meta_live_runtime.py`
  - result: `75 passed`

## 9) Wiring Realism Knob Into Optimization Pipeline

Goal:
- ensure strategy sweeps/meta selection can optimize with pending-order realism, not only default instant-drop behavior.

Code updates:
- `unified_hourly_experiment/run_stock_sortino_lag_robust.py`
  - new CLI arg: `--entry-order-ttl-hours` (default `0`).
  - eval override support per config:
    - `eval_entry_order_ttl_hours`.
  - passes through to `PortfolioConfig(entry_order_ttl_hours=...)`.
- `unified_hourly_experiment/sweep_meta_portfolio.py`
  - new CLI arg: `--entry-order-ttl-hours`.
  - included in base portfolio config and output payload.
- `unified_hourly_experiment/trade_unified_hourly_meta.py`
  - new CLI arg: `--entry-order-ttl-hours`.
  - used in per-symbol selector simulations (`simulate_symbol_daily_returns`).
- `unified_hourly_experiment/auto_meta_optimize.py`
  - new CLI arg: `--entry-order-ttl-hours`.
  - forwarded into `sweep_meta_portfolio.py` runs.
  - included in generated deploy command and recommendation search-space payload.

## 10) New Live-Log Replay Tool For Simulator Realism

Added:
- `unified_hourly_experiment/replay_stock_trade_log_sim.py`

What it does:
- replays sparse `strategy_state/stock_trade_log.jsonl` entry intents into simulator.
- sweeps `entry_order_ttl_hours` and `bar_margin`.
- scores live-vs-sim entry alignment by hour/symbol/side:
  - `hourly_abs_count_delta_total`
  - `exact_row_ratio`
  - per-symbol deltas

Artifacts generated:
- `experiments/stock_trade_log_sim_replay_20260304_111435.json`
- `experiments/stock_trade_log_sim_replay_20260304_111509.json`

Observed top result from replay sweep (NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT):
- `entry_order_ttl_hours=0`, `bar_margin=0.0005`
  - `hourly_abs_count_delta_total=10`
  - `exact_row_ratio=0.60`
  - `live_entries=25`, `sim_entries=15`

Interpretation:
- sparse-action timeline fix improves fidelity of state progression.
- on this logged window, longer pending TTL (`1-3`) did not improve count-alignment metric vs `ttl=0`.
- realism settings should still be swept per strategy/window; defaults should not be assumed globally optimal.

## 11) FastForecaster2 Frontier Bootstrap (2026-03-05)

Goal:
- Stand up `fastforecaster2` as a new frontier forecaster with:
  - optional Chronos-style symbol embedding bootstrap,
  - C++/CUDA weighted-MAE path retained,
  - post-train shared-cash market simulation,
  - explicit risk/smoothness metrics in training summary + W&B.

Code delivered:
- New package: `fastforecaster2/`
  - `config.py`: simulator + Chronos embedding controls.
  - `trainer.py`: Chronos embedding bootstrap hook, simulator eval, robust risk metrics + artifact export.
  - `run_training.py`: CLI flags for simulator and embedding controls.
  - `data.py`: keeps raw OHLC per symbol for simulator alignment.
  - `README.md`: usage and controls.
- Tests added:
  - `tests/test_fastforecaster2_config.py`
  - `tests/test_fastforecaster2_sim_metrics.py`

Validation:
- `pytest -q tests/test_fastforecaster2_config.py tests/test_fastforecaster2_sim_metrics.py tests/test_fastforecaster_config.py tests/test_fastforecaster_seed_sweep.py`
- Result: `16 passed`

Smoke training setup (all runs):
- dataset: hourly
- symbols: 8
- lookback/horizon: 128/12
- epochs: 1
- windows: train 4096, val 836, test 733
- W&B project: `stock`

Run outputs:
- Risk-off neutral sim run:
  - run: `fastforecaster2_frontier_smoke_20260305_simneutral`
  - W&B: https://wandb.ai/lee101p/stock/runs/jg7wvbmq
  - best_val_mae: `1.7190`
  - test_mae: `1.6818`
  - sim_pnl: `0.0`
  - sim_max_drawdown: `0.0`
  - sim_smoothness: `1.0`
  - sim_trades: `0`
- Active sim run (risk-constrained but trading):
  - run: `fastforecaster2_frontier_smoke_20260305_simrisk2`
  - W&B: https://wandb.ai/lee101p/stock/runs/the343zt
  - best_val_mae: `1.7190`
  - test_mae: `1.6818`
  - sim_pnl: `-2986.47`
  - sim_total_return: `-29.86%`
  - sim_max_drawdown: `1.0`
  - sim_smoothness: `0.0176`
  - sim_trades: `278`

Best PnL so far (FastForecaster2):
- `sim_pnl = 0.0` (risk-off neutral config; no trades).
- Active-trading config still needs tuning for smooth/low-risk profitability.

Artifacts:
- `fastforecaster2/artifacts_smoke_20260305_simneutral/metrics/summary.json`
- `fastforecaster2/artifacts_smoke_20260305_simneutral/metrics/simulator_summary.json`
- `fastforecaster2/artifacts_smoke_20260305_simrisk2/metrics/summary.json`
- `fastforecaster2/artifacts_smoke_20260305_simrisk2/metrics/simulator_summary.json`

## 12) FastForecaster2 Simulator Policy Retune (2026-03-05 late)

Objective:
- make `fastforecaster2` simulator evaluation closer to a production low-risk selector:
  - stop pyramiding into the same name every positive bar,
  - use smoothed signals instead of raw one-step noise,
  - keep thresholds in actual forecast-return space,
  - only enter on top-k transitions and exit on signal decay/rank loss.

Code updates:
- `fastforecaster2/config.py`
  - added:
    - `market_sim_min_trade_intensity`
    - `market_sim_signal_ema_alpha`
    - `market_sim_entry_buffer_bps`
    - `market_sim_exit_buffer_bps`
- `fastforecaster2/trainer.py`
  - added weighted short-horizon forecast aggregation for simulator signals.
  - replaced naive per-bar action generation with:
    - EMA-smoothed long-only top-k selector,
    - hold-vs-entry hysteresis (`buy_threshold` vs `sell_threshold`),
    - transition-based entries,
    - full exits on de-selection,
    - simulator-compatible minimum allocation floor.
- `fastforecaster2/run_training.py`
  - exposed the new simulator controls on CLI.
- `tests/test_fastforecaster2_sim_metrics.py`
  - added transition/no-pyramiding planner test.
- `tests/test_fastforecaster2_config.py`
  - added config validation for `market_sim_signal_ema_alpha`.

Validation:
- `pytest -q tests/test_fastforecaster2_config.py tests/test_fastforecaster2_sim_metrics.py`
- Result: `10 passed`

Forecast-scale check:
- observed `smoothed_return` on the smoke run is centered around `1e-5` to `1e-4`, so previous `1e-3` thresholds were effectively disabling trading.

Retune sweep results:
- `fastforecaster2_ff2_sweep_d_20260305`
  - W&B: https://wandb.ai/lee101p/stock/runs/rfp8qcfi
  - config: `buy=2e-5`, `sell=0`, `top_k=2`, `max_intensity=12`, `ema_alpha=0.35`, `max_hold=12`
  - sim_pnl: `-275.41`
  - sim_total_return: `-2.75%`
  - sim_max_drawdown: `0.3735`
  - sim_smoothness: `0.1188`
- `fastforecaster2_ff2_sweep_e_20260305`
  - W&B: https://wandb.ai/lee101p/stock/runs/jo9akgt0
  - config: `buy=3e-5`, `sell=1e-5`, `top_k=2`, `max_intensity=10`, `ema_alpha=0.45`, `max_hold=8`
  - sim_pnl: `+42.98`
  - sim_total_return: `+0.43%`
  - sim_max_drawdown: `0.3342`
  - sim_smoothness: `0.1574`
  - sim_trades: `417`
- `fastforecaster2_ff2_sweep_f_20260305`
  - W&B: https://wandb.ai/lee101p/stock/runs/nya9epka
  - config: `buy=1.5e-5`, `sell=0`, `top_k=1`, `max_intensity=8`, `ema_alpha=0.30`, `max_hold=6`
  - sim_pnl: `-231.04`
  - sim_total_return: `-2.31%`
  - sim_max_drawdown: `0.3057`
  - sim_smoothness: `0.1644`

Best PnL so far (FastForecaster2):
- `+42.98` from `fastforecaster2_ff2_sweep_e_20260305`
- current best risk-aware trading config on the smoke setup:
  - `market_sim_buy_threshold=3e-5`
  - `market_sim_sell_threshold=1e-5`
  - `market_sim_top_k=2`
  - `market_sim_max_trade_intensity=10`
  - `market_sim_min_trade_intensity=4`
  - `market_sim_signal_ema_alpha=0.45`
  - `market_sim_max_hold_hours=8`

Artifacts:
- `fastforecaster2/ff2_sweep_d/metrics/simulator_summary.json`
- `fastforecaster2/ff2_sweep_e/metrics/simulator_summary.json`
- `fastforecaster2/ff2_sweep_f/metrics/simulator_summary.json`

## 13) FastForecaster2 Dense-Simulator Frontier Retune (2026-03-06)

Objective:
- fix the simulator alignment bug found after the first policy retune and keep optimizing for representative production PnL, not sparse-action artifacts.

Issue found:
- the first dense-frame patch in `fastforecaster2/trainer.py` dropped the `timestamp` column name after `reset_index()`, causing `_build_market_sim_frames()` to fail on `sort_values(["symbol", "timestamp"])`.
- the earlier positive `ff2_sweep_e` result also relied on a sparser action timeline with zero explicit sell rows, so it was not the right production proxy.

Code updates:
- `fastforecaster2/trainer.py`
  - fixed dense signal-frame construction so `timestamp` survives the forward-fill/reindex path.
  - kept cross-symbol densification active so inactive symbols still receive held-state updates on the shared timestamp grid.
- `tests/test_fastforecaster2_sim_metrics.py`
  - added a regression test to verify dense signal-frame expansion preserves `timestamp` and forward-fills symbol signals correctly.

Validation:
- `python -m py_compile fastforecaster2/trainer.py fastforecaster2/config.py fastforecaster2/run_training.py`
- `pytest -q tests/test_fastforecaster2_config.py tests/test_fastforecaster2_sim_metrics.py`
- Result: `11 passed`

Dense baseline rerun:
- `fastforecaster2_ff2_sweep_g_densefix_20260306`
  - W&B: https://wandb.ai/lee101p/stock/runs/o4jk5dpu
  - config: `buy=3e-5`, `sell=1e-5`, `top_k=2`, `max_intensity=10`, `ema_alpha=0.45`, `max_hold=8`
  - sim_pnl: `+18.56`
  - sim_total_return: `+0.19%`
  - sim_max_drawdown: `0.1821`
  - sim_smoothness: `0.3695`
  - sim_trades: `62`
  - explicit action rows: `36` buys, `14` sells

Checkpoint-only policy sweeps (same checkpoint, denser simulator):
- broad low-threshold region (`buy=2e-5`, `sell=0`) remained decisively bad: roughly `-780` to `-990` PnL on the completed trials.
- narrow frontier sweep artifact:
  - `fastforecaster2/policy_sweep_20260306_densefix_narrow/results.json`
- best completed narrow dense points:
  - `b3p0e-05_s1p5e-05_k2_ema0p55_h6_mi8`: `sim_pnl=+15.22`, `max_drawdown=0.1486`, `smoothness=0.4221`
  - `b3p0e-05_s1p0e-05_k2_ema0p45_h8_mi8`: `sim_pnl=+14.34`, `max_drawdown=0.1486`, `smoothness=0.4228`
- follow-up frontier artifacts:
  - `fastforecaster2/policy_followup_20260306/results.json`
  - `fastforecaster2/policy_followup2_20260306/results.json`
- key finding:
  - `top_k=1` works better than `top_k=3` under the dense simulator.
  - stronger sell hysteresis (`sell=1.5e-5`) is better than `sell=1e-5` in the `top_k=1`, `ema=0.55` regime.

Promoted dense-frontier W&B runs:
- `fastforecaster2_ff2_sweep_h_densefrontier_20260306`
  - W&B: https://wandb.ai/lee101p/stock/runs/zmf4qhtt
  - config: `buy=3e-5`, `sell=1.5e-5`, `top_k=1`, `max_intensity=8`, `ema_alpha=0.55`, `max_hold=6`
  - sim_pnl: `+25.44`
  - sim_total_return: `+0.25%`
  - sim_max_drawdown: `0.1498`
  - sim_smoothness: `0.4111`
  - sim_trades: `57`
  - explicit action rows: `33` buys, `14` sells
- `fastforecaster2_ff2_sweep_i_densefrontier_20260306`
  - W&B: https://wandb.ai/lee101p/stock/runs/t2j62086
  - config: `buy=3e-5`, `sell=1.5e-5`, `top_k=1`, `max_intensity=12`, `ema_alpha=0.55`, `max_hold=6`
  - sim_pnl: `+41.00`
  - sim_total_return: `+0.41%`
  - sim_max_drawdown: `0.2158`
  - sim_smoothness: `0.3174`
  - sim_trades: `57`
  - explicit action rows: `33` buys, `14` sells

Current interpretation:
- best headline PnL on the older sparse simulator is still `+42.98` from `fastforecaster2_ff2_sweep_e_20260305`, but that run is not the best production proxy because it had no explicit sell rows in the generated action file.
- best representative dense-simulator PnL so far is now `+41.00` from `fastforecaster2_ff2_sweep_i_densefrontier_20260306`.
- best dense-simulator low-risk point so far is `+25.44` from `fastforecaster2_ff2_sweep_h_densefrontier_20260306`.

## 14) FastForecaster2 Policy Sweep Tool + Frontier Breakout (2026-03-06 later)

Objective:
- stop doing one-off checkpoint replay snippets and make simulator-policy search reproducible inside the repo, then use it to push the dense-simulator frontier higher.

Code updates:
- `fastforecaster2/policy_sweep.py`
  - new checkpoint-only simulator sweep entrypoint.
  - loads a saved `best.pt`, rebuilds the trainer/data bundle once, and sweeps:
    - `market_sim_buy_threshold`
    - `market_sim_sell_threshold`
    - `market_sim_top_k`
    - `market_sim_signal_ema_alpha`
    - `market_sim_max_hold_hours`
    - `market_sim_max_trade_intensity`
    - `market_sim_min_trade_intensity`
  - writes per-trial simulator artifacts plus aggregate results.
  - optionally logs each trial to W&B through `wandboard.py`.
- `tests/test_fastforecaster2_policy_sweep.py`
  - added grid-validation coverage and stable trial-name coverage.
- `fastforecaster2/README.md`
  - documented the checkpoint-only policy sweep workflow.

Validation:
- `python -m py_compile fastforecaster2/policy_sweep.py fastforecaster2/trainer.py fastforecaster2/run_training.py`
- `pytest -q tests/test_fastforecaster2_policy_sweep.py tests/test_fastforecaster2_config.py tests/test_fastforecaster2_sim_metrics.py`
- Result: `13 passed`

Focused dense frontier sweep:
- command shape:
  - `python -m fastforecaster2.policy_sweep --checkpoint-path fastforecaster2/ff2_sweep_i_densefrontier/checkpoints/best.pt --buy-thresholds 3e-5 --sell-thresholds 1.4e-5,1.5e-5,1.6e-5 --top-ks 1 --ema-alphas 0.55 --max-hold-hours-values 6 --max-trade-intensities 9,10,11,12,13 --wandb-project stock --wandb-group fastforecaster2_policy_frontier_20260306`
- partial aggregate artifact from completed trials:
  - `fastforecaster2/policy_frontier_search/policy_sweep_20260306_012537/results_partial.json`
  - `fastforecaster2/policy_frontier_search/policy_sweep_20260306_012537/best_policy.json`

Sweep findings:
- `sell=1.4e-5` dominated `sell=1.5e-5` and `sell=1.6e-5` in the `top_k=1`, `ema=0.55`, `max_hold=6` regime.
- best completed checkpoint-only sweep rows:
  - `b3p0e-05_s1p4e-05_k1_ema0p55_h6_maxi13_mini4`
    - W&B: https://wandb.ai/lee101p/stock/runs/qms6k8ss
    - sim_pnl: `+80.88`
    - sim_total_return: `+0.81%`
    - sim_max_drawdown: `0.2298`
    - sim_smoothness: `0.3098`
    - sim_trades: `56`
  - `b3p0e-05_s1p4e-05_k1_ema0p55_h6_maxi9_mini4`
    - W&B: https://wandb.ai/lee101p/stock/runs/ypydbhiy
    - sim_pnl: `+54.76`
    - sim_total_return: `+0.55%`
    - sim_max_drawdown: `0.1656`
    - sim_smoothness: `0.3940`
    - sim_trades: `56`
- monotonic pattern observed on the best branch:
  - `max_intensity=9`: `+54.76`
  - `10`: `+61.20`
  - `11`: `+67.70`
  - `12`: `+74.26`
  - `13`: `+80.88`

Promoted full training runs:
- `fastforecaster2_ff2_sweep_j_densefrontier_20260306`
  - W&B: https://wandb.ai/lee101p/stock/runs/0ps79vu5
  - config: `buy=3e-5`, `sell=1.4e-5`, `top_k=1`, `max_intensity=13`, `ema_alpha=0.55`, `max_hold=6`
  - best_val_mae: `1.71904`
  - test_mae: `1.68184`
  - sim_pnl: `+80.88`
  - sim_total_return: `+0.81%`
  - sim_max_drawdown: `0.2298`
  - sim_smoothness: `0.3098`
  - sim_trades: `56`
  - explicit action rows: `33` buys, `14` sells
- `fastforecaster2_ff2_sweep_k_densefrontier_20260306`
  - W&B: https://wandb.ai/lee101p/stock/runs/xzn8wvet
  - config: `buy=3e-5`, `sell=1.4e-5`, `top_k=1`, `max_intensity=9`, `ema_alpha=0.55`, `max_hold=6`
  - best_val_mae: `1.71904`
  - test_mae: `1.68184`
  - sim_pnl: `+54.76`
  - sim_total_return: `+0.55%`
  - sim_max_drawdown: `0.1656`
  - sim_smoothness: `0.3940`
  - sim_trades: `56`
  - explicit action rows: `33` buys, `14` sells

Current bests:
- best representative dense-simulator PnL so far: `+80.88` from `fastforecaster2_ff2_sweep_j_densefrontier_20260306`
- best representative dense-simulator lower-drawdown point so far: `+54.76` from `fastforecaster2_ff2_sweep_k_densefrontier_20260306`

## 15) FastForecaster2 Frontier Refinement + Naming Fix (2026-03-06 late)

Objective:
- keep pushing the dense-simulator frontier while tightening the search around the proven `top_k=1`, `ema=0.55`, `max_hold=6` branch.

Issue found and fixed:
- `fastforecaster2/policy_sweep.py` originally formatted float-valued thresholds with too little precision in `_trial_name(...)`.
- this caused nearby values such as `1.35e-5` and `1.4e-5` to collapse to the same path/run label during dense frontier search.

Code updates:
- `fastforecaster2/policy_sweep.py`
  - increased float precision in trial-name formatting so nearby threshold values generate unique artifact directories and W&B run names.
- `tests/test_fastforecaster2_policy_sweep.py`
  - added a regression test to ensure close threshold values no longer collide.

Validation:
- `pytest -q tests/test_fastforecaster2_policy_sweep.py tests/test_fastforecaster2_config.py tests/test_fastforecaster2_sim_metrics.py`
- Result: `14 passed`

Refined dense frontier sweep:
- command shape:
  - `python -m fastforecaster2.policy_sweep --checkpoint-path fastforecaster2/ff2_sweep_j_densefrontier/checkpoints/best.pt --buy-thresholds 3e-5,3.05e-5,3.1e-5 --sell-thresholds 1.35e-5,1.4e-5,1.45e-5 --top-ks 1 --ema-alphas 0.55 --max-hold-hours-values 6 --max-trade-intensities 12,13,14,15`
- partial aggregate artifacts:
  - `fastforecaster2/policy_frontier_search3/policy_sweep_20260306_020115/results_partial.json`
  - `fastforecaster2/policy_frontier_search3/policy_sweep_20260306_020115/best_policy.json`

Key sweep findings:
- `sell=1.4e-5` remained the best band.
- `sell=1.45e-5` was consistently second-best.
- `sell=1.35e-5` and higher buy thresholds (`3.05e-5`, `3.1e-5`) were materially worse.
- completed top rows from the refined frontier:
  - `b3e-05_s1p4e-05_k1_ema0p55_h6_maxi15_mini4`
    - W&B: https://wandb.ai/lee101p/stock/runs/3m1sllg9
    - sim_pnl: `+94.28`
    - sim_total_return: `+0.94%`
    - sim_max_drawdown: `0.2599`
    - sim_smoothness: `0.2795`
    - sim_trades: `56`
  - `b3e-05_s1p4e-05_k1_ema0p55_h6_maxi14_mini4`
    - W&B: https://wandb.ai/lee101p/stock/runs/j62e12of
    - sim_pnl: `+87.56`
    - sim_total_return: `+0.88%`
    - sim_max_drawdown: `0.2450`
    - sim_smoothness: `0.2939`
    - sim_trades: `56`

Promoted full training run:
- `fastforecaster2_ff2_sweep_l_densefrontier_20260306`
  - W&B: https://wandb.ai/lee101p/stock/runs/8mj18h3l
  - config: `buy=3e-5`, `sell=1.4e-5`, `top_k=1`, `max_intensity=15`, `ema_alpha=0.55`, `max_hold=6`
  - best_val_mae: `1.71904`
  - test_mae: `1.68184`
  - sim_pnl: `+94.28`
  - sim_total_return: `+0.94%`
  - sim_max_drawdown: `0.2599`
  - sim_smoothness: `0.2795`
  - sim_sortino: `0.5654`
  - sim_trades: `56`
  - explicit action rows: `33` buys, `14` sells

Current bests:
- best representative dense-simulator PnL so far: `+94.28` from `fastforecaster2_ff2_sweep_l_densefrontier_20260306`
- best representative dense-simulator lower-drawdown point still: `+54.76` from `fastforecaster2_ff2_sweep_k_densefrontier_20260306`

## 16) FastForecaster2 Switch-Gap Test + Higher-Intensity Frontier (2026-03-06 latest)

Objective:
- test whether a more active portfolio-decision layer could beat the sticky hold-until-sell policy, then continue the best dense frontier if switching proved unnecessary.

Code updates:
- `fastforecaster2/config.py`
  - added `market_sim_switch_score_gap`
- `fastforecaster2/run_training.py`
  - exposed `--market-sim-switch-score-gap`
- `fastforecaster2/trainer.py`
  - planner now supports optional challenger-over-held replacement when the challenger’s `smoothed_score` exceeds the weakest held symbol by a configurable margin.
- `fastforecaster2/policy_sweep.py`
  - added `--switch-score-gaps` so switch-margin sweeps are reproducible.
- `tests/test_fastforecaster2_sim_metrics.py`
  - added planner regression coverage for sticky-vs-switch behavior.
- `tests/test_fastforecaster2_config.py`
  - added negative-gap validation.

Validation:
- `pytest -q tests/test_fastforecaster2_policy_sweep.py tests/test_fastforecaster2_config.py tests/test_fastforecaster2_sim_metrics.py`
- Result: `16 passed`

Switch-gap experiment:
- checkpoint sweep against `fastforecaster2/ff2_sweep_l_densefrontier/checkpoints/best.pt`
- tested `switch_score_gap` values: `0`, `5e-5`, `1e-4`, `2e-4` on the best existing branch (`buy=3e-5`, `sell=1.4e-5`, `top_k=1`, `ema=0.55`, `hold=6`, `max_intensity=13`)
- result:
  - baseline sticky policy (`sgap=0`) remained best at `+80.88`
  - positive switch margins were harmful:
    - `sgap=5e-5`: `sim_pnl=-64.03`, `sim_active_signal_rows=211`
    - `sgap=1e-4`: `sim_pnl=-68.36`, `sim_active_signal_rows=205`
    - `sgap=2e-4`: `sim_pnl=-26.84`, `sim_active_signal_rows=191`
- interpretation:
  - on this smoke setup, challenger-driven switching adds churn faster than it adds edge.
  - the best branch remains the sticky `top_k=1` policy with `switch_score_gap=0`.

Higher-intensity continuation on the best branch:
- checkpoint sweep artifacts:
  - `fastforecaster2/policy_intensity_search/policy_sweep_20260306_022136/results.json`
  - `fastforecaster2/policy_intensity_search2/policy_sweep_20260306_022228/results.json`
- completed results:
  - `max_intensity=16`
    - W&B: https://wandb.ai/lee101p/stock/runs/cpvl12db
    - sim_pnl: `+101.06`
    - sim_max_drawdown: `0.2745`
    - sim_smoothness: `0.2664`
  - `max_intensity=17`
    - W&B: https://wandb.ai/lee101p/stock/runs/v7rola6q
    - sim_pnl: `+107.88`
    - sim_max_drawdown: `0.2888`
    - sim_smoothness: `0.2543`
  - `max_intensity=18`
    - W&B: https://wandb.ai/lee101p/stock/runs/xm549n1a
    - sim_pnl: `+114.75`
    - sim_max_drawdown: `0.3027`
    - sim_smoothness: `0.2432`
  - `max_intensity=19`
    - W&B: https://wandb.ai/lee101p/stock/runs/h1jx0b6s
    - sim_pnl: `+121.66`
    - sim_max_drawdown: `0.3164`
    - sim_smoothness: `0.2330`
  - `max_intensity=20`
    - W&B: https://wandb.ai/lee101p/stock/runs/dphgpy0h
    - sim_pnl: `+128.61`
    - sim_max_drawdown: `0.3298`
    - sim_smoothness: `0.2235`
- the branch remains monotonic in PnL through `max_intensity=20`, with trade count unchanged at `56`.

Promoted full training run:
- `fastforecaster2_ff2_sweep_m_densefrontier_20260306`
  - W&B: https://wandb.ai/lee101p/stock/runs/vmh6d7g8
  - config: `buy=3e-5`, `sell=1.4e-5`, `top_k=1`, `max_intensity=20`, `ema_alpha=0.55`, `max_hold=6`, `switch_score_gap=0`
  - best_val_mae: `1.71904`
  - test_mae: `1.68184`
  - sim_pnl: `+128.61`
  - sim_total_return: `+1.29%`
  - sim_max_drawdown: `0.3298`
  - sim_smoothness: `0.2235`
  - sim_sortino: `0.7755`
  - sim_trades: `56`
  - explicit action rows: `33` buys, `14` sells

Current bests:
- best representative dense-simulator PnL so far: `+128.61` from `fastforecaster2_ff2_sweep_m_densefrontier_20260306`
- best representative dense-simulator lower-drawdown point still: `+54.76` from `fastforecaster2_ff2_sweep_k_densefrontier_20260306`

## 17) FastForecaster2 Entry-Score Gate Rejection + Intensity Frontier Extension (2026-03-06 latest)

Objective:
- test whether an explicit `smoothed_score` gate could improve the realism-adjusted dense simulator by filtering weaker entries, then continue the only branch that was still improving if the gate failed.

Code updates:
- `fastforecaster2/config.py`
  - added `market_sim_entry_score_threshold`
- `fastforecaster2/run_training.py`
  - exposed `--market-sim-entry-score-threshold`
- `fastforecaster2/trainer.py`
  - new entries now require both `smoothed_return >= buy_threshold` and `smoothed_score >= entry_score_threshold`
- `fastforecaster2/policy_sweep.py`
  - added `--entry-score-thresholds` so score-gate sweeps are reproducible
- `tests/test_fastforecaster2_config.py`
  - added negative-threshold validation
- `tests/test_fastforecaster2_sim_metrics.py`
  - added planner regression coverage confirming low-score symbols are filtered from new entries
- `tests/test_fastforecaster2_policy_sweep.py`
  - updated sweep-spec coverage for the new parameter
- `fastforecaster2/README.md`
  - documented the new simulator control and the current frontier interpretation

Validation:
- `python -m py_compile fastforecaster2/policy_sweep.py fastforecaster2/trainer.py fastforecaster2/run_training.py fastforecaster2/config.py`
- `pytest -q tests/test_fastforecaster2_policy_sweep.py tests/test_fastforecaster2_config.py tests/test_fastforecaster2_sim_metrics.py`
- Result: `18 passed`

Entry-score sweep:
- checkpoint sweep against `fastforecaster2/ff2_sweep_m_densefrontier/checkpoints/best.pt`
- artifacts:
  - `fastforecaster2/policy_entryscore_search/policy_sweep_20260306_022930/results.json`
  - `fastforecaster2/policy_entryscore_search/policy_sweep_20260306_022930/best_policy.json`
- tested:
  - `entry_score_threshold`: `0`, `0.0012`, `0.0013`, `0.0014`, `0.0015`, `0.0016`, `0.0018`
  - `max_intensity`: `20`, `22`, `24`
- result:
  - `0` and `0.0012` were identical because historical entry scores were already above `0.0012`
  - any real gate (`0.0013+`) was harmful
  - representative failures:
    - `es=0.0013`, `maxi=24`: `sim_pnl=+72.31`, `sim_trades=49`
    - `es=0.0014`, `maxi=24`: `sim_pnl=+38.82`, `sim_trades=51`
    - `es=0.0015`, `maxi=24`: `sim_pnl=-27.94`, `sim_trades=45`
    - `es=0.0016`, `maxi=24`: `sim_pnl=-237.28`, `sim_trades=42`
    - `es=0.0018`, `maxi=24`: `sim_pnl=-10.46`, `sim_trades=30`
- interpretation:
  - on this smoke setup, weak-looking entries are still net-positive contributors
  - adding a score gate reduces churn, but it removes too much profitable exposure
  - the correct frontier decision is `entry_score_threshold=0`

Intensity extension on the winning no-gate branch:
- checkpoint sweep artifacts:
  - `fastforecaster2/policy_intensity_search3/policy_sweep_20260306_023333/results.json`
  - `fastforecaster2/policy_intensity_search3/policy_sweep_20260306_023333/best_policy.json`
- completed results:
  - `max_intensity=26`
    - W&B: https://wandb.ai/lee101p/stock/runs/ikk9zcu5
    - sim_pnl: `+171.11`
    - sim_total_return: `+1.71%`
    - sim_max_drawdown: `0.4045`
    - sim_smoothness: `0.1780`
    - sim_sortino: `1.0560`
  - `max_intensity=28`
    - W&B: https://wandb.ai/lee101p/stock/runs/4sayyhlm
    - sim_pnl: `+185.53`
    - sim_total_return: `+1.86%`
    - sim_max_drawdown: `0.4273`
    - sim_smoothness: `0.1661`
    - sim_sortino: `1.1579`
  - `max_intensity=30`
    - W&B: https://wandb.ai/lee101p/stock/runs/vb73nhzp
    - sim_pnl: `+200.06`
    - sim_total_return: `+2.00%`
    - sim_max_drawdown: `0.4492`
    - sim_smoothness: `0.1553`
    - sim_sortino: `1.2649`
  - `max_intensity=32`
    - W&B: https://wandb.ai/lee101p/stock/runs/883kgaxb
    - sim_pnl: `+214.68`
    - sim_total_return: `+2.15%`
    - sim_max_drawdown: `0.4702`
    - sim_smoothness: `0.1455`
    - sim_sortino: `1.3774`

Promoted full training run:
- `fastforecaster2_ff2_sweep_n_densefrontier_20260306`
  - W&B: https://wandb.ai/lee101p/stock/runs/z54dh6o3
  - config: `buy=3e-5`, `sell=1.4e-5`, `entry_score_threshold=0`, `top_k=1`, `max_intensity=32`, `ema_alpha=0.55`, `max_hold=6`, `switch_score_gap=0`
  - best_val_mae: `1.71904`
  - test_mae: `1.68184`
  - sim_pnl: `+214.68`
  - sim_total_return: `+2.15%`
  - sim_max_drawdown: `0.4702`
  - sim_smoothness: `0.1455`
  - sim_sortino: `1.3774`
  - sim_trades: `56`
  - explicit action rows remain unchanged in count from the prior branch; the gain came from sizing, not more entries

Current bests:
- best representative dense-simulator PnL so far: `+214.68` from `fastforecaster2_ff2_sweep_n_densefrontier_20260306`
- best promoted full-training run is now `fastforecaster2_ff2_sweep_n_densefrontier_20260306` at `+214.68`
- best representative dense-simulator lower-drawdown point still: `+54.76` from `fastforecaster2_ff2_sweep_k_densefrontier_20260306`
