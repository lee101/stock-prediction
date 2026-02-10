# RL Trading Experiment

End-to-end RL for crypto hourly trading using C market simulator and PPO.

## Architecture
- C market simulator (csim/) with PufferLib-style API
- 3 symbols: BTC, ETH, SOL (FDUSD zero-fee pairs)
- Action space: Discrete(7) - hold, buy/sell each symbol
- Observation: 10 features/symbol + position info = 36 dim
- MLP policy: 2x256 hidden layers, 77K params
- PPO training with GAE

## Results (70-day validation, $10K initial)

| Model | Return | Max DD | Sortino | Notes |
|-------|--------|--------|---------|-------|
| Supervised (ft30 selector) | 42.5x | -1.1% | 123.9 | With work-steal |
| RL v1 (10M steps, 256h) | 0.80x | -42% | -1.05 | 3e-4 lr, 168 ep |
| RL v2 (50M steps, 512h) | 0.88x | -36% | -0.60 | 1e-4 lr, 336 ep |

## Why RL underperforms
1. Limited features: 10 basic price features vs supervised model's 96-step transformer + Chronos2 forecasts
2. No temporal context: MLP sees single bar vs transformer seeing 96 bars
3. Reward signal: sparse equity changes don't guide exploration well
4. Training data: ~32K training bars is small for RL exploration

## Build
```
cd csim && make test        # C unit tests
cd .. && python3 setup.py build_ext --inplace  # Build binding
python3 -m rl_trading.train --help
python3 -m rl_trading.evaluate --checkpoint checkpoints/best.pt
```
