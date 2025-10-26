# Guard Evaluation Automation

## Daily Checklist
1. `python evaltests/run_guard_backtests.py` – refresh standard high-sample guard runs (produces JSON, summaries, scoreboard updates).
2. `python evaltests/update_guard_history.py` – append latest baseline metrics to `guard_compile_history.json` (use `--config` / `--variant` for alternate compile configs).
3. `python evaltests/render_compile_history.py` – regenerate `guard_compile_history.md` and the enriched stats table (`guard_compile_stats.md`).
4. `python evaltests/compare_high_samples.py` and `python evaltests/compare_compile_modes.py` – refresh comparison markdown tables (pass `--compile-suffix _real_full_compile128.json` to capture the baseline-sample diagnostic run).

## Optional (low GPU window)
- `python evaltests/run_guard_backtests.py --config evaltests/guard_backtest_targets_compile.json` – gather compile-enabled sweeps for GOOG/META/TSLA.
- Follow with steps 2–4 above to capture history/markdown updates.
2. `python evaltests/run_guard_backtests.py --config evaltests/guard_backtest_targets_compile.json` – optional compile sweep (during low GPU usage).
3. `python evaltests/update_guard_history.py --config evaltests/guard_backtest_targets_compile.json --variant compile` – append latest baseline vs compile metrics to `guard_compile_history.json`.
4. `python evaltests/render_compile_history.py` – regenerate `guard_compile_history.md` plus `guard_compile_stats.md` (means, sign counts, heuristics).
5. `python evaltests/compare_high_samples.py` and `python evaltests/compare_compile_modes.py` – refresh comparison markdown tables.
6. `python evaltests/run_guard_backtests.py --config evaltests/guard_backtest_targets_compile128.json` – compile sweep with baseline sampling (diagnostic run for regression triage).
7. `python evaltests/update_guard_history.py --config evaltests/guard_backtest_targets_compile128.json --variant compile128` – log the baseline-sample compile metrics.
8. `python evaltests/compare_compile_modes.py --compile-suffix _real_full_compile128.json --output evaltests/guard_compile_comparison_compile128.md` – emit markdown for the baseline-sample compile comparison.

## Key Artifacts
- `evaltests/guard_metrics_summary.md` – merged guard telemetry (validation, hold-out, backtests).
- `evaltests/guard_vs_baseline.md` – MaxDiff vs simple strategy deltas (mock + real + high-sample).
- `evaltests/guard_highsample_comparison.md` – baseline vs high-sample deltas.
- `evaltests/guard_compile_comparison.md` – baseline vs compile deltas (latest window).
- `evaltests/guard_compile_history.json/md` – historical record of compile vs baseline metrics.

## Promotion Criteria (draft)
- High-sample: require ≥3 consecutive runs with positive MaxDiff uplift and neutral-to-lower val loss per symbol.
- Compile: require stable positive deltas (or meaningful val-loss reductions) across ≥3 windows before moving settings into `guard_backtest_targets.json`.
