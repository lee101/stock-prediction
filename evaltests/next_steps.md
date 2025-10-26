# RL Triage Snapshot (2025-10-24)

- **DeepSeek baselines** remain the clear leaders (net PnL ≈ $6.65 and Sharpe ≈ +0.62), setting an upper bound for current fully-automated RL stacks.
- **GymRL sweeps** now deliver positive validation returns (loss-probe v10: +10.6% cumulative, avg daily +0.0051, turnover 0.15) with the first positive Sharpe proxy, but hold-out variance remains high.
- **PufferLib pipeline (TC=5 bps, risk penalty 0.05)** marginally improves AMZN_MSFT pair (best val profit 0.0037) but still trails DeepSeek; consider optuna sweep on risk penalty, leverage limit, and specialist learning rates.
- **Differentiable Market risk sweep** (risk_aversion 0.25, drawdown λ 0.05) mildly improves Sharpe (−0.434 vs −0.452) but total return remains negative; further reward-tuning required (e.g., positive wealth objective, variance penalty on weights).

## Suggested Next Experiments
1. **GymRL PPO**  
   - Loss-shutdown v11 maintains positive Sharpe (~+0.00016) with turnover 0.155; let pipeline cool briefly, then run a lightweight confirmation sweep (turnover penalty ≈0.0071, loss probe 0.002) and compare against v10/v11 logs.  
   - Regime guard calibration (drawdown 3.6 %, trailing return −3 %, turnover 0.55, probe 0.002, leverage scale 0.6) trims turnover −0.008 and lowers leverage to 0.48× on the stressed window while leaving earlier slices (start indices 3 600/3 300) essentially unchanged—guards now only trigger in adverse regimes.  
   - Guard-aware confirmation sweep (`gymrl_confirmation_guarded_v12`) completed: validation cumulative return +10.96%, guard turnover hit rate ≈4.8%, drawdown/negative guards dormant. Hold-out stress slice shows guards firing (drawdown 45%, negative 40%) with turnover collapsing to 0.066 and leverage scale ≈0.82; other slices (0–3000) show minimal guard activity. Mock backtests now cover AAPL/NVDA/GOOG/TSLA/META (see `evaltests/backtests/gymrl_guard_confirm_{symbol}.json`).  
   - **New:** Live (non-mock) backtests now cover the full basket (AAPL/GOOG/META/NVDA/TSLA) with dynamic Toto OOM handling; MaxDiff beats simple by +12.6 pts on average. JSON export is part of the run (see `gymrl_guard_confirm_{symbol}_real_full*.json`); high-sample presets (512–4096 Toto samples) are rolled out for GOOG/META/TSLA. Compile trials (GOOG/META/TSLA) completed without OOMs, but gains are small—use `python evaltests/run_guard_backtests.py --config evaltests/guard_backtest_targets_compile.json` during off-peak windows and monitor `guard_compile_comparison.md` before promoting compile as the default.
   - **Action:** Latest compile sweep (2025-10-24T20:27Z) tanked GOOG simple return (Δ −0.1105) while META/TSLA continue to flip signs. Diagnostic rerun with compile + baseline sampling (128) confirms GOOG simple return still collapses (Δ −0.163), META drifts −0.0015, and TSLA improves +0.1005. Capture Toto compile traces/latency for GOOG next, then bisect META/TSLA with targeted instrumentation before any rollout.

2. **PufferLib Portfolio Stage**  
   - Run focused Optuna sweep across `risk_penalty` 0.02–0.08, `leverage_limit` 1.2–1.6, and RL learning rate 1e-4–5e-4.  
   - Track pair-level Sharpe and cumulative return, targeting positive AMZN_MSFT performance.

3. **Differentiable Market GRPO**  
   - Switch wealth objective to Sharpe, raise `variance_penalty_mode='weights'`, and test `risk_aversion` {0.35, 0.5}.  
   - Evaluate 2022–2024 windows to ensure robustness before rerunning 2024–2025 windows.

Status: queued experiments completed (`evaltests/run_queue.json`); awaiting new queue after decisions above.
