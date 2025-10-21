# Differentiable Market Kronos

`differentiable_market_kronos` extends the base differentiable market RL trainer with
frozen Kronos embeddings. A pre-trained Kronos tokenizer/model encodes rolling
windows of OHLCV data to provide rich context features while the downstream RL
policy remains fully differentiable.

## Key Features

- Frozen Kronos tokenizer/model: no gradient updates during RL training.
- Configurable context length, embedding mode (`context`, `bits`, `both`), and
  batching device used for Kronos inference.
- Seamless reuse of the differentiable market environment, policy, and
  evaluation pipeline; only the feature builder is replaced.

## Quick Start

```bash
uv sync
source .venv/bin/activate
python -m differentiable_market_kronos.train \
  --data-root trainingdata \
  --lookback 192 \
  --kronos-model NeoQuasar/Kronos-small \
  --kronos-tokenizer NeoQuasar/Kronos-Tokenizer-base \
  --kronos-context 192 \
  --save-dir differentiable_market_kronos/runs
```

Use `--kronos-device cuda` to run Kronos embedding extraction on GPU while the
policy can remain on a different device if needed.

## Testing

```bash
source .venv/bin/activate
pytest tests/differentiable_market_kronos -q
```

The tests monkeypatch Kronos components with lightweight stubs so they remain
fast and deterministic.
