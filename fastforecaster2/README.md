# fastforecaster2

`fastforecaster2` is a frontier experiment built from `FastForecaster` with:
- MAE-first forecasting training (hourly or daily).
- Optional C++/CUDA weighted-MAE kernels.
- Optional Chronos-style symbol embedding bootstrap.
- Post-train shared-cash market simulation with risk metrics.
- A smoothed long-only top-k simulator policy that enters on rank transitions and exits on signal decay.
- WandBoard/W&B logging for train, validation, and simulator outcomes.

## Quick run

```bash
source .venv313/bin/activate
python -m fastforecaster2.run_training \
  --dataset hourly \
  --epochs 3 \
  --max-symbols 12 \
  --wandb-project stock
```

## Chronos embedding bootstrap (optional)

Pass a symbol->vector mapping file (`.json`, `.pt`, `.npy`, `.npz`):

```bash
source .venv313/bin/activate
python -m fastforecaster2.run_training \
  --dataset hourly \
  --chronos-embeddings-path experiments/chronos_symbol_embeddings.json \
  --chronos-embeddings-blend 0.25
```

## Market simulator controls

`fastforecaster2` runs a post-train market simulation by default and writes:
- `metrics/simulator_summary.json`
- `metrics/simulator_equity.csv`
- `metrics/simulator_actions.csv`

Key flags:
- `--no-market-sim-eval`
- `--market-sim-top-k`
- `--market-sim-max-trade-intensity`
- `--market-sim-min-trade-intensity`
- `--market-sim-buy-threshold`
- `--market-sim-sell-threshold`
- `--market-sim-entry-score-threshold`
- `--market-sim-signal-ema-alpha`
- `--market-sim-switch-score-gap`
- `--market-sim-entry-buffer-bps`
- `--market-sim-exit-buffer-bps`
- `--market-sim-max-hold-hours`

Notes:
- Forecast return magnitudes are small on this setup, so useful thresholds are often around `1e-5` to `1e-4`, not `1e-3`.
- The simulator planner is intentionally long-only and uses transition-based entries to avoid pyramiding on every positive bar.

## Checkpoint-only policy sweep

Use this to retune simulator policy knobs against a saved checkpoint without retraining:

```bash
source .venv313/bin/activate
python -m fastforecaster2.policy_sweep \
  --checkpoint-path fastforecaster2/ff2_sweep_i_densefrontier/checkpoints/best.pt \
  --buy-thresholds 3e-5 \
  --sell-thresholds 1.4e-5,1.5e-5,1.6e-5 \
  --entry-score-thresholds 0,0.0013,0.0015 \
  --top-ks 1 \
  --ema-alphas 0.55 \
  --max-hold-hours-values 6 \
  --max-trade-intensities 9,10,11,12,13 \
  --switch-score-gaps 0,5e-5,1e-4 \
  --wandb-project stock
```

Outputs:
- `results.json`
- `best_policy.json`
- per-trial `metrics/simulator_summary.json`

Current smoke-frontier note:
- The best dense-simulator branch so far is still `top_k=1`, `ema=0.55`, `buy=3e-5`, `sell=1.4e-5`, `switch_score_gap=0`, `entry_score_threshold=0`.
- On the current 8-symbol smoke setup, increasing `market_sim_max_trade_intensity` improved PnL monotonically through `32`, while positive `entry_score_threshold` values reduced PnL.
