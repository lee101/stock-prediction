# rl-trainingbinance

Risk-aware hourly RL experiments for Binance-style multi-asset trading.

This directory is a self-contained experiment path for:

- hourly Binance symbol baskets
- long/short trading with configurable shortability
- up to 5x gross leverage
- train-only feature normalization carried through to checkpoints
- reward shaping that prefers smoother PnL over raw aggression
- purge-aware walk-forward validation over multiple horizons
- batched validation across windows for faster 7d / 30d sweeps
- grid sweeps over leverage and risk hyperparameters

It is deliberately lightweight and explicit. The default backbone is a small
transformer encoder over recent hourly features, which is easier to iterate on
than trying to wire a full Qwen stack directly into numeric trading data.

## Default basket

The default symbol preset is:

- `BTCFDUSD`
- `ETHFDUSD`
- `AAVEFDUSD`
- `DOGEUSD`
- `SUIUSDT`
- `SOLFDUSD`

Assumption:

- `USD`, `USDT`, and `FDUSD` are treated as near-par stable quotes for this
  experiment path.
- `BTCFDUSD` and `ETHFDUSD` get zero trading fees by default.

## Train

```bash
source .venv313/bin/activate
python rl-trainingbinance/train.py \
  --output-dir analysis/rl_trainingbinance_smoke \
  --episode-steps 720 \
  --updates 10 \
  --validate-every 2 \
  --early-stop-patience 3 \
  --num-envs 4 \
  --rollout-steps 32 \
  --validation-batch-size 8 \
  --validation-window-hours 168,720 \
  --validation-window-weights 0.4,0.6
```

## Validate

```bash
source .venv313/bin/activate
python rl-trainingbinance/validate.py \
  --checkpoint analysis/rl_trainingbinance_smoke/best.pt \
  --batch-size 8 \
  --window-hours 168,720 \
  --window-weights 0.4,0.6
```

## Sweep

```bash
source .venv313/bin/activate
python rl-trainingbinance/sweep.py \
  --output-root analysis/rl_trainingbinance_sweep \
  --limit 4 \
  -- --episode-steps 720 --updates 6 --validate-every 2 --early-stop-patience 3 --num-envs 4 --rollout-steps 16 \
     --validation-batch-size 8 --validation-window-hours 168,720 --validation-window-weights 0.4,0.6
```

## Outputs

Training writes:

- `best.pt`
- `final.pt`
- `manifest.json`
- `history.json`
- feature normalization stats and multi-horizon validation summaries in manifests/checkpoints
- `ranking.csv` / `ranking.json` from sweeps

Validation prints and optionally saves:

- aggregate weighted score across horizons
- per-horizon 7d / 30d summaries
- p10 total return
- median Sortino
- p90 max drawdown
- turnover / volatility summaries
- a conservative composite score
