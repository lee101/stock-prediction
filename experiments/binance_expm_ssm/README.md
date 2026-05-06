# Binance Matrix-Exponential SSM

Differentiable continuous-time state-space model for market regimes. The model
learns a generator `Q_t`, computes exact transition probabilities with
`torch.matrix_exp(Q_t)`, and maps the resulting regime distribution to portfolio
weights.

Quick smoke run:

```bash
python experiments/binance_expm_ssm/expm_ssm.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 720 \
  --eval-hours 168 \
  --epochs 10
```

Research purpose:

- Tests the "learn Q with a neural net, exponentiate for exact transitions"
  idea without Monte Carlo sampling.
- Keeps gradients through `torch.matrix_exp`, so it can later become an actor,
  critic, or differentiable risk model.
- Still needs hard binary-fill pufferlib validation before any production use.
