# RL Triage Snapshot (2025-10-23)

- **DeepSeek baselines** remain the clear leaders (net PnL ≈ $6.65 and Sharpe ≈ +0.62), setting an upper bound for current fully-automated RL stacks.
- **GymRL sweep (turnover penalty 0.001)** still posts negative validation return (−9.3%) with very high turnover (0.65) and max intraday leverage above 2×. Reward shaping needs additional downside pressure (e.g., stronger turnover/L2 penalties or leverage interest).
- **PufferLib pipeline (TC=5 bps, risk penalty 0.05)** marginally improves AMZN_MSFT pair (best val profit 0.0037) but still trails DeepSeek; consider optuna sweep on risk penalty, leverage limit, and specialist learning rates.
- **Differentiable Market risk sweep** (risk_aversion 0.25, drawdown λ 0.05) mildly improves Sharpe (−0.434 vs −0.452) but total return remains negative; further reward-tuning required (e.g., positive wealth objective, variance penalty on weights).

## Suggested Next Experiments
1. **GymRL PPO**  
   - Loss-shutdown v5 now at +11.7% cumulative (Sharpe ≈ −0.0061); next iteration should test `turnover_penalty=0.005`, consider smaller `loss_shutdown_probe_weight` (0.01), and explore zero-entropy final stage.  
   - Feature cache alignment for hold-out remains unresolved; options: resample CSVs to common hour or narrow to symbols with identical timestamp cadence.

2. **PufferLib Portfolio Stage**  
   - Run focused Optuna sweep across `risk_penalty` 0.02–0.08, `leverage_limit` 1.2–1.6, and RL learning rate 1e-4–5e-4.  
   - Track pair-level Sharpe and cumulative return, targeting positive AMZN_MSFT performance.

3. **Differentiable Market GRPO**  
   - Switch wealth objective to Sharpe, raise `variance_penalty_mode='weights'`, and test `risk_aversion` {0.35, 0.5}.  
   - Evaluate 2022–2024 windows to ensure robustness before rerunning 2024–2025 windows.

Status: queued experiments completed (`evaltests/run_queue.json`); awaiting new queue after decisions above.
