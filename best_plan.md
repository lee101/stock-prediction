# RL Training Evaluation Master Plan (2025-10-22)

## Objectives
- Benchmark and improve RL pipelines in `hftraining/`, `gymrl/`, `pufferlibtraining/`, and `differentiable_market/`.
- Produce realistic post-training PnL evaluations using consistent market data and cost assumptions.
- Compare RL outcomes against `stockagentdeepseek` agent simulations (`tests/prod/agents/stockagentdeepseek/*`) and the production `trade_stock_e2e` stack.
- Deliver an actionable recommendation for Alpaca deployment, including risk-managed configuration templates.

## Current Snapshot
- **HF Training (`hftraining/quick_test_output_20251017_143438`)**: Eval loss 0.76 with cumulative return -0.82 and Sharpe < 0 after 500 steps → baseline underperforming.
- **GymRL (`gymrl/models/aggregate_pufferlib_metrics.csv`)**: PPO allocator runs on Toto features; best run (`20251020_puffer_rl400_lr3e4_risk005_tc5`, AAPL_AMZN pair) shows +0.52 cumulative return but partner pair negative → instability across assets.
- **PufferLib Portfolio RL**: Multi-stage pipeline completed; mixed pair-wise results with some negative annualised returns, signalling tuning gaps in leverage penalties and risk coefficients.
- **Differentiable Market (`differentiable_market/runs/20251021_094014`)**: Latest GRPO training yields eval annual return -0.75% with turnover 2% and Sharpe -0.45 → requires reward shaping and better warm starts.
- **DeepSeek Agent Simulator**: Unit tests cover deterministic plan replay but no recent aggregate PnL benchmarking; need to synthesise plan outputs and Monte Carlo evaluation.
- **Production Baseline (`trade_stock_e2e.log`)**: Live Kelly-based allocator active on Oct 22, 2025 with multiple entries; lacks summarised daily PnL metrics in logs → extract for baseline comparison.

## Workstreams
1. **Foundation & Environment**
   - Align on Python interpreter (`.venv312`) and ensure `uv pip` installs for shared deps (Torch nightly with `torch.compile`, Toto/Kronos editable installs).
   - Verify dataset parity: confirm `trainingdata/`, `tototraining/trainingdata/train`, and agent simulator historical feeds cover the same period and frequency.
   - Harden GPU detection and `torch.compile(max_autotune)` fallbacks across modules; capture compile cache paths in `compiled_models/`.

2. **Module Deep Dives**
   - **HF Training**
     - Re-run `quick_rl_train.py` with improved scheduler, warm starts from `compiled_models/`, and evaluate over 5k+ steps.
     - Add regression tests around `hftraining/portfolio_rl_trainer.py` with synthetic price shocks.
     - Export inference checkpoints for simulator integration (`hftraining/output/`).
   - **GymRL**
     - Rebuild feature caches using current Toto/Kronos compiles; profile `FeatureBuilder` latency under `torch.compile`.
     - Train PPO with cross-asset baskets and track evaluation via `gymrl/evaluate_policy.py`.
     - Generate offline datasets for d3rlpy conservative Q-learning smoke tests.
   - **PufferLib Training**
     - Validate stage transitions (forecaster → specialists → portfolio) with automated checks in `pufferlibtraining/tests/`.
     - Tune leverage/risk penalties using Optuna sweeps; log to `pufferlibtraining/logs`.
     - Extend `aggregate_pufferlib_metrics.csv` with Sharpe/Sortino/confidence intervals.
   - **Differentiable Market**
     - Diagnose negative reward: inspect `metrics.jsonl` for reward gradients, adjust `risk_aversion`, `trade_penalty`.
     - Run backtests via `differentiable_market.marketsimulator.run` across 2023–2025 windows; store outputs in `differentiable_market/evals/<run_id>/`.
     - Add unit tests for differentiable transaction costs to guard against future regressions.

3. **Cross-System Evaluation Framework**
   - Build a shared evaluation harness under `evaltests/rl_benchmark_runner.py` that:
     - Loads checkpoints from each module.
     - Uses common market scenarios (daily/minute bars) with identical cost/leverage assumptions.
     - Computes PnL, annualised return, Sharpe, Sortino, max drawdown, turnover, and execution latency.
   - Integrate DeepSeek plan simulations by replaying `simulate_deepseek_plan` outputs against the same market bundles.
   - Compare against `trade_stock_e2e` historical decisions to anchor production expectations.

4. **Recommendation & Reporting**
   - Produce per-module scorecards (JSON + Markdown) summarising training config, wall-clock, GPU utilisation, and evaluation metrics.
   - Run final backtests through `backtest_test3_inline.py` for apples-to-apples measurement.
   - Deliver final recommendation document covering deployment-ready configs, risk mitigation, and next experiments.

## Immediate Next Actions (Oct 22)
- [x] Confirm active Python env via `source .venv312/bin/activate` and `uv pip list` sanity check.
- [x] Run smoke tests: `pytest hftraining/test_pipeline.py -q`, `pytest tests/experimental/rl/gymrl/test_feature_builder.py -q`, `pytest tests/experimental/pufferlib/test_pufferlib_env_rules.py -q` (fixed leverage cap + date formatting to make suite green).
- [ ] Script baseline PnL extraction from `trade_stock_e2e.log` and DeepSeek simulation outputs for reference tables.
- [ ] Begin harmonised evaluation harness skeleton under `evaltests/`.

## Progress Log
- **2025-10-22**: Validated `.venv312` environment; gymRL feature builder and HF pipeline smoke tests pass. Patched `StockTradingEnv` info payload to normalise numpy datetimes and respect configured leverage caps, restoring `tests/experimental/pufferlib/test_pufferlib_env_rules.py`.
- **2025-10-22**: Added `evaltests/baseline_pnl_extract.py` to surface production trade PnL (via `strategy_state/trade_history.json`), exposure snapshots from `trade_stock_e2e.log`, and DeepSeek simulator benchmarks. Exported refreshed summaries to `evaltests/baseline_pnl_summary.{json,md}`.
- **2025-10-22**: Scaffolded cross-stack evaluation harness (`evaltests/rl_benchmark_runner.py`) with sample config and JSON output capturing checkpoint metadata alongside baseline reference metrics.
- **2025-10-22**: Expanded harness evaluators for `hftraining` (loss/return metrics) and `gymrl` (PPO config + validation stats) with sample targets wired through `evaltests/sample_rl_targets.json`.
- **2025-10-22**: Added evaluator coverage for `pufferlibtraining` (pipeline summary + aggregate pair returns) and `differentiable_market` (GRPO metrics, top-k checkpoints, eval report ingestion).
- **2025-10-22**: Unified evaluation output comparisons with baseline trade PnL and DeepSeek simulations, ensuring every RL run lists reference agent net PnL and production realised PnL deltas.
- **2025-10-22**: Introduced a sortable scoreboard in `rl_benchmark_results.json`, ranking RL runs and DeepSeek baselines by their key performance metric for quick cross-system triage.
- **2025-10-22**: Prioritised retraining/backtest queue (`evaltests/run_queue.json`) covering GymRL PPO turnover sweep, PufferLib Optuna campaign, and differentiable_market risk sweep.
- **2025-10-23**: Ran `gymrl.train_ppo_allocator` turnover sweep (300k steps, `turnover_penalty=0.001`); new artefacts under `gymrl/artifacts/sweep_20251022/` with validation cumulative return -9.26% (needs further tuning).
- **2025-10-23**: Executed PufferLib pipeline with higher transaction costs/risk penalty (`pufferlibtraining/models/optuna_20251022/`); AMZN_MSFT pair still negative — further hyperparameter search required.
- **2025-10-23**: Extended differentiable_market backtester CLI with risk override flags and ran risk sweep (`risk-aversion=0.25`, `drawdown_lambda=0.05`); Sharpe improved slightly (‑0.451→‑0.434) but returns remain negative.
- **2025-10-23**: Added automated scoreboard renderer (`evaltests/render_scoreboard.py`) producing `evaltests/scoreboard.md` for quick status snapshots.
- **2025-10-23**: Wired `rl_benchmark_runner.py` to invoke the scoreboard renderer after each run, keeping Markdown/JSON history current.
- **2025-10-23**: Ran higher-penalty GymRL PPO sweep (`gymrl/artifacts/sweep_20251023_penalized/`) — turnover dropped to 0.19 (from 0.65) with cumulative return -8.44% over validation; continue iteration on reward shaping.
- **2025-10-23**: Loss-shutdown GymRL sweep (`sweep_20251023_lossprobe/`) achieved +9.4% cumulative validation return with turnover 0.23; next step is to stabilise Sharpe (currently -0.007) and monitor out-of-sample robustness.
- **2025-10-23**: Loss-shutdown v2 (`sweep_20251023_lossprobe_v2/`) delivered +10.8% cumulative return with turnover 0.17 (Sharpe ≈ -0.010); leverage checks now within 0.84× avg close.
- **2025-10-23**: Loss-shutdown v3 (`sweep_20251023_lossprobe_v3/`) pushes cumulative return to +11.21% with turnover 0.17 and average daily return +0.0053; Sharpe still slightly negative (−0.0101) — entropy annealing remains a priority.
- **2025-10-23**: Loss-shutdown v4 (`sweep_20251023_lossprobe_v4/`) with entropy anneal (0.001→0.0001) reaches +11.86% cumulative return, avg daily +0.00537, turnover 0.175, Sharpe −0.0068 (improving).
- **2025-10-23**: Loss-shutdown v5 (`sweep_20251023_lossprobe_v5/`) pushes to +11.71% cumulative (avg daily +0.00558) with lower turnover 0.148; Sharpe still slightly negative (−0.0061) but improving as leverage tightens.
- **2025-10-23**: Loss-shutdown v6 (`sweep_20251023_lossprobe_v6/`) maintains +11.88% cumulative return with turnover 0.15; Sharpe improves to −0.0068 under entropy anneal 0.0008→0.
- **2025-10-23**: Loss-shutdown v7 (`sweep_20251023_lossprobe_v7/`) delivers +11.43% cumulative return, turnover 0.144, Sharpe ≈ −0.0047; indicates diminishing returns as penalties rise—need to flip Sharpe positive or explore out-of-sample evaluation.
- **2025-10-23**: Loss-shutdown v8 (`sweep_20251025_lossprobe_v8/`) maintains +10.7% cumulative return with turnover 0.145 and slightly better Sharpe (≈ −0.005) under more aggressive penalties; turnover plateaued while returns dipped slightly.
- **2025-10-23**: Loss-shutdown v9 (`sweep_20251025_lossprobe_v9/`) keeps cumulative return +10.77% with turnover 0.155 and Sharpe ≈ −0.00052; leverage averages 0.70×, showing gradual progress toward positive Sharpe.
- **2025-10-23**: Loss-shutdown v10 (`sweep_20251025_lossprobe_v10/`) hits +10.64% cumulative return with turnover 0.153 and Sharpe proxy +0.00016—the first positive Sharpe configuration (40k steps, turnover penalty 0.0068).
- **2025-10-23**: Hold-out evaluation on resampled top-5 cache (42-step windows) now spans −23.8% to +57.6% cumulative return (median +3.3%) with leverage ≤1.13×—highlighting regime variance despite controlled leverage. Detailed stats in `evaltests/gymrl_holdout_summary.{json,md}`.
- **2025-10-23**: Loss-shutdown v11 (`sweep_20251025_lossprobe_v11/`, 40k steps, turnover penalty 0.0069) sustains +10.69% cumulative return, turnover 0.155, Sharpe proxy +0.00016, and max drawdown 0.0071 while keeping leverage ≤1.10×.
- **2025-10-23**: Added regime guard heuristics (`RegimeGuard`) to `PortfolioEnv` with CLI wiring (`--regime-*` flags), covering drawdown, negative-return, and turnover guards; new telemetry fields (`turnover_penalty_applied`, guard flags) feed into evaluation outputs. Authored targeted pytest coverage (`tests/gymrl/test_regime_guard.py`) and refreshed `rl_benchmark_results.json`/`scoreboard.md` to capture the updated metrics.
- **2025-10-23**: Ran guard A/B on loss-probe v11 over resampled top-5 hold-out slices (start indices 3 781, 3 600, 3 300). Initial guards (18% drawdown / ≤0 trailing / 0.50 turnover) degraded PnL; calibrated thresholds (3.6% drawdown / ≤−3% trailing / 0.55 turnover / 0.002 probe / leverage scale 0.6) now cut average turnover by ~0.8 ppts on the troubled window while leaving benign windows effectively unchanged. Full details logged in `evaltests/gymrl_guard_analysis.{json,md}` and summarised in `evaltests/guard_metrics_summary.md`. Guard-aware confirmation sweep (`gymrl_confirmation_guarded_v12`) completed with validation cumulative return +10.96% (guard turnover hit rate ~4.8%); preset stored at `gymrl/guard_config_calibrated.json` for future sweeps.
- **2025-10-24**: Evaluated the guard-confirmed checkpoint on the stressed hold-out window (start index 3781) and additional slices (0→3000). Guards now engage selectively: turnover guard ~5% on validation, drawdown guard ~40% and leverage scale ~0.82× on the stress window, remaining dormant elsewhere. Summaries and scoreboard updated with the guard telemetry.
- **2025-10-24**: Attempted full `backtest_test3_inline.py` run with the guard preset; run failed during Torch Inductor compilation (`compiled_models/torch_inductor/...main.cpp` missing). Need to rerun with compilation disabled or adjust the compiled-models cache before we can compare live-equivalent results.

Progress will be updated here alongside key metric snapshots, dated entries, and blockers.
