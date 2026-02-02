# Binance Progress Log

Updated: 2026-02-03

## Notes / Fixes
- Added `state_dict`-aware max_len inference when loading classic models so positional encoding buffers match checkpoint shapes.
  - Updated: `binanceneural/model.py`, `binancechronossolexperiment/inference.py`, `binanceneural/run_simulation.py`, `binanceneural/trade_binance_hourly.py`, `binanceneural/sweep.py`.
- New PnL/RL utility library (`src/tradinglib/`) with tests (metrics, rewards, benchmarks, target gating).

## Experiments (Chronos2 + Binance neural)

### Baseline (classic, full history)
- Run: `chronos_sol_20260202_041139`
- Metrics (10-day test): total_return=0.3433135070, sortino=136.8685975259, annualized_return=49893.15079393116
- Last 2 days: total_return=0.08457621078, sortino=218.4418841462, annualized_return=2722471.909152695

### Nano + Muon (90d history)
- Run: `chronos_sol_nano_muon_20260202_230000`
- Command:
  `python -m binancechronossolexperiment.run_experiment --symbol SOLUSDT --data-root trainingdatahourlybinance --forecast-cache-root binancechronossolexperiment/forecast_cache --model-id chronos2_finetuned/SOLUSDT_lora_20260202_030749/finetuned-ckpt --context-hours 3072 --chronos-batch-size 32 --horizons 1 --sequence-length 72 --val-days 20 --test-days 10 --max-history-days 90 --epochs 6 --batch-size 128 --learning-rate 3e-4 --weight-decay 1e-4 --optimizer muon_mix --model-arch nano --num-kv-heads 2 --mlp-ratio 4.0 --logits-softcap 12.0 --muon-lr 0.02 --muon-momentum 0.95 --muon-ns-steps 5 --cache-only --no-compile --run-name chronos_sol_nano_muon_20260202_230000`
- Metrics (10-day test): total_return=0.1933922754, sortino=50.0260710990, annualized_return=651.0995876932
- Last 2 days: total_return=0.045477, sortino=70.023416, annualized_return=3347.674857

### Nano + AdamW (90d history)
- Run: `chronos_sol_nano_adamw_20260202_230500`
- Metrics (10-day test): total_return=0.2216511106, sortino=47.5818631247, annualized_return=1536.6274827896
- Last 2 days: total_return=0.044726, sortino=46.942293, annualized_return=2936.213771

### Nano + Muon (365d history)
- Run: `chronos_sol_nano_muon_20260202_235000`
- Metrics (10-day test): total_return=0.178760, sortino=33.205916, annualized_return=413.887796
- Last 2 days: total_return=0.057223, sortino=166.957110, annualized_return=25727.928457

### Classic + Muon (365d history, retry after pos-encoding fix)
- Run: `chronos_sol_classic_muon_20260202_235500_retry`
- Metrics (10-day test): total_return=0.131389, sortino=38.260542, annualized_return=91.263087
- Last 2 days: total_return=-0.002678, sortino=-2.415733, annualized_return=-0.386949

### Nano + Muon (full history)
- Run: `chronos_sol_nano_muon_full_20260203_001000`
- Metrics (10-day test): total_return=0.163166, sortino=133.889350, annualized_return=253.654864
- Last 2 days: total_return=0.049308, sortino=3301.882512, annualized_return=6527.535363

### Nano + AdamW (full history)
- Run: `chronos_sol_nano_adamw_full_20260203_003000`
- Metrics (10-day test): total_return=0.131405, sortino=106.950196, annualized_return=91.309086
- Last 2 days: total_return=0.039455, sortino=388.584324, annualized_return=1165.838243

### Nano + Muon (multi-horizon 1/4/24, full history)
- Run: `chronos_sol_nano_muon_h1_4_24_20260203_010000`
- Metrics (10-day test): total_return=0.139216, sortino=92.137741, annualized_return=117.786575
- Last 2 days: total_return=0.029039, sortino=1656.626735, annualized_return=184.700605

### Nano + AdamW (multi-horizon 1/4/24, full history)
- Run: `chronos_sol_nano_adamw_h1_4_24_20260203_092000`
- Metrics (10-day test): total_return=0.150219, sortino=139.825174, annualized_return=167.955148
- Last 2 days: total_return=0.056290, sortino=1120.777238, annualized_return=21899.280913

## RL Reward Sweep (SOLUSDT hourly)
- Run: `python -m rlsys.run_reward_sweep --csv trainingdatahourlybinance/SOLUSDT.csv`
- Results saved: `rlsys/results/reward_sweep.json`
- Raw reward: episode_reward=-0.026216, episode_sharpe=-2.740867, episode_max_drawdown=-0.199934
- Risk-adjusted reward: episode_reward=-0.061468, episode_sharpe=-2.304354, episode_max_drawdown=-0.222763
- Sharpe-like reward: episode_reward=-0.115716, episode_sharpe=-2.304354, episode_max_drawdown=-0.222763

## Live Binance
- Attempted single-cycle live run failed due to restricted-region Binance API eligibility.
- Supervisor `binanceexp1-solusd` now running with BINANCE_TLD=us; requires Binance.US API keys (current keys rejected with "API-key format invalid").
- See `binanceresults.md` for full details of live attempt and command used.

## Next Experiments
- Longer RL sweeps (e.g., 50k+ timesteps) with a grid over drawdown/volatility penalties.
- Multi-horizon (1/4/24) variants with larger sequence length (96/128) and/or context_hours 2048 vs 3072.
- Run longer backtests (30d+) on the best multi-horizon run and log trade stats + PnL targets.

## Binanceexp1 SOLUSD (marketsimulator, hourly)

Data & validation
- Backfilled hourly gap using Alpaca: 2025-04-16 through 2025-12-21.
- Max gap after backfill: ~3 hours.

Checkpoint
- `binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt`

Intensity sweep (price_offset_pct=0, horizon=24, sequence_length=96, min_gap_pct=0.0003)
- 1.0 -> total_return 7.8325, sortino 55.8105
- 1.2 -> total_return 8.3542
- 1.6 -> total_return 9.0235
- 1.8 -> total_return 9.3230
- 2.0 -> total_return 9.5891
- 2.2 -> total_return 9.8570
- 2.5 -> total_return 10.1909
- 3.0 -> total_return 10.6377
- 4.0 -> total_return 11.3114
- 6.0 -> total_return 12.4451
- 8.0 -> total_return 13.5462
- 10.0 -> total_return 14.4308
- 15.0 -> total_return 16.2541, sortino 66.6059
- 20.0 -> total_return 17.7449, sortino 66.1148 (best PnL so far)

Risk mitigation tests
- Probe-mode after loss: reduced total return vs baseline.
- Max-hold forced close: reduced total return vs baseline.

Production target (SOLUSD)
- intensity-scale 20.0, horizon 24, sequence length 96, min-gap-pct 0.0003
- Runner: `scripts/run_binanceexp1_prod_solusd.sh`
- Supervisor: `supervisor/binanceexp1-solusd.conf`
