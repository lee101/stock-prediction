# Daily Policy, Hourly Replay (2026-02-10)

Purpose: take the best daily-trained `pufferlib_market` policy and replay its **daily action trace** on **hourly bars** to compute higher-resolution risk metrics and count order executions (helps catch "double execution"/thrash artifacts).

## Repro

```bash
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

python -m pufferlib_market.replay_eval \
  --checkpoint experiments/pufferlib_market_daily_20260210/checkpoints/daily_mix8_h256_lr3e4_ent001_noanneal_2M/best.pt \
  --daily-data-path experiments/pufferlib_market_daily_20260210/eval50d_mktd_v2.bin \
  --hourly-data-root trainingdatahourly \
  --start-date 2025-12-17 --end-date 2026-02-05 \
  --max-steps 50 --fee-rate 0.001 --max-leverage 1.0 \
  --deterministic \
  --arch mlp --hidden-size 256 \
  --output-json experiments/pufferlib_market_daily_hourly_replay_20260210/replay_report.json
```

The JSON output is printed to stdout and written to `replay_report.json` if `--output-json` is provided.

