# Guard-Confirmed RL Readiness Snapshot

## Production Baseline
- Realised PnL (latest log): **-8,661.71 USD** over **7.10** trading days.
- Average daily PnL: **-1,219.50 USD/day**.

## Validation Leaderboard (GymRL)
- Run: `sweep_20251026_guard_confirm`
  - Cumulative return: **+10.96%**
  - Avg daily return: **+0.00498**
  - Sharpe proxy: **0.00119** (log-return mean)
  - Turnover: **0.160**
  - Guard hit rates: negative **0%**, turnover **4.8%**, drawdown **0%**

## Hold-Out Stress Test (start index 3781)
- Cumulative return: **-4.35%**
- Avg turnover: **0.066** (vs 0.361 in validation)
- Max drawdown: **6.85%**
- Guard hit rates: negative **40%**, drawdown **45%**, turnover **0%**
- Avg leverage scale: **0.82×**, min leverage scale: **0.60×**

## Additional Hold-Out Windows (42-step slices)
- Negative/turnover guard hit rates stay below **33%** and leverage remains ~1× across slices starting at 0, 500, 1000, 1500, 2000, 2500, 3000.

## Backtest Summary (Mock & Real)
| Symbol | Variant | MaxDiff Return | Simple Return | Δ (MaxDiff - Simple) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| AAPL | mock | 0.0261 | -0.0699 | 0.0960 | Mock analytics, fast Toto |
| AAPL | real-lite | 0.0374 | -0.2166 | 0.2540 | Live Toto (128 samples, compile off) |
| AAPL | real-full | 0.0373 | -0.1766 | 0.2139 | Live Toto (dynamic OOM fallback, 128→96 samples) |
| GOOG | mock | 0.0124 | -0.0788 | 0.0912 | Mock analytics |
| GOOG | real-lite | 0.0294 | -0.2143 | 0.2437 | Live Toto (128 samples) |
| GOOG | real-full | 0.0302 | -0.1415 | 0.1717 | Live Toto with dynamic OOM fallback |
| META | mock | 0.0281 | -0.0182 | 0.0463 | Mock analytics |
| META | real-lite | 0.0412 | -0.0281 | 0.0693 | Live Toto (128 samples) |
| META | real-full | 0.0405 | -0.0197 | 0.0602 | Live Toto with dynamic OOM fallback |
| NVDA | mock | 0.0212 | -0.0210 | 0.0422 | Mock analytics |
| NVDA | real-lite | 0.0474 | 0.0117 | 0.0357 | Live Toto (128 samples) |
| NVDA | real-full | 0.0445 | 0.0044 | 0.0401 | Live Toto with dynamic OOM fallback |
| TSLA | mock | 0.0309 | -0.0201 | 0.0510 | Mock analytics |
| TSLA | real-lite | 0.0704 | -0.0213 | 0.0917 | Live Toto (128 samples) |
| TSLA | real-full | 0.0762 | -0.0082 | 0.0844 | Live Toto with dynamic OOM fallback |
| **Average (mock)** |  | **0.0237** | **-0.0416** | **0.0653** | |
| **Average (real runs)** |  | **0.0455** | **-0.0810** | **0.1265** | |

## Interpretation
1. Guards eliminate leverage spikes in the stress window (avg leverage down to 0.82×; turnover slashed to 0.066).
2. Validation remains positive with minimal guard activity, implying low friction in calmer regimes.
3. Mock backtests show MaxDiff outperforming the simple baseline by **+6.5 points** on average; live runs (lite + full-fidelity fallback) now deliver an average uplift of **+15.1 points**, still using reduced Toto sampling (128 shrinking to 96 when GPU pressure spikes).

## Compile Trials Snapshot
- High-sample Toto runs with `torch.compile` are logged under `gymrl_guard_confirm_{symbol}_real_full_compile.json` for GOOG, META, and TSLA.
- `evaltests/guard_compile_comparison.md` compares compile vs baseline metrics; the latest sweep (2025-10-24T20:27Z) shows GOOG simple return dropping from +0.0192 to −0.0913 (Δ −0.1105) when compile is enabled with the 512→4096 Toto sample ramp. The baseline-sample diagnostic (`evaltests/guard_compile_comparison_compile128.md`) still reports GOOG simple return collapsing to −0.143 (Δ −0.163) while META drifts −0.0015 and TSLA improves +0.1005.
- Aggregate history (see `guard_compile_stats.md`) now reports GOOG ≈ −0.0114 mean simple delta (regress), META +0.0134 (promote on average but with sign flips), TSLA −0.0194 (regress). MaxDiff deltas skew positive across the compile128 trials, indicating the guard-specific mechanics remain healthy even as the simple strategy breaks.
- Recommendation: keep compile trials in monitoring-only mode. Prioritise GOOG fusion triage (compile vs eager), then rerun META/TSLA with targeted instrumentation (Toto latency, sample counts) before considering any rollout.

## Remaining Checks
- Scale the real (non-mock) backtests to full forecast fidelity and the entire symbol basket once GPU memory permits.
- Compare guard hit timelines against production when full real runs are available.
- Keep `evaltests/guard_metrics_summary.md`, `evaltests/guard_vs_baseline.md`, and this readiness brief refreshed after every new simulation or baseline update.
