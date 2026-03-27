# E2E Chronos2 Stock Training

`e2etraining/` is a stock-only proof-of-concept for training a shared Chronos2 backbone
and a portfolio policy head end to end.

Current scope:

- daily stock data from `trainingdata/`
- stock universe sourced from `available_stocks_with_data.json`
- shared Chronos2 backbone across all selected symbols
- optional LoRA on Chronos2 attention projections
- portfolio head that maps Chronos quantile outputs into cross-sectional weights
- differentiable objective combining:
  - portfolio log-growth
  - Sortino-style reward
  - drawdown penalty
  - auxiliary forecast pinball loss

This is deliberately narrower than the full production stack. The goal is to prove that
RL gradients can update Chronos-adapter weights across a broad stock universe before
adding more realistic execution logic.

Example smoke run:

```bash
source .venv313/bin/activate
python -m e2etraining.train \
  --include-symbols AAPL,MSFT,NVDA,AMZN,GOOG,META,AMD,TSLA \
  --max-assets 8 \
  --min-timesteps 512 \
  --context-length 64 \
  --rollout-length 4 \
  --batch-size 1 \
  --steps 1 \
  --eval-every 1 \
  --device cuda \
  --run-name smoke_local
```

Larger research run:

```bash
source .venv313/bin/activate
python -m e2etraining.train \
  --max-assets 64 \
  --min-timesteps 1024 \
  --context-length 128 \
  --rollout-length 16 \
  --batch-size 4 \
  --steps 200 \
  --eval-every 20 \
  --cross-learning \
  --device cuda \
  --run-name stock64_proof
```
