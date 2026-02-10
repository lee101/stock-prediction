# Binance Failed/Underperforming Experiments

## RL Trading (C Simulator + PPO) - 2026-02-08

**Goal**: End-to-end RL for crypto hourly trading. Replace supervised pipeline with direct reward optimization.

**Setup**:
- C market simulator with PufferLib-style API (csim/trading_sim.h)
- 3 symbols: BTCUSD, ETHUSD, SOLUSD on FDUSD zero-fee pairs
- Action space: Discrete(7) - hold, buy/sell each symbol
- Observation: 10 basic price features per symbol + position info = 36 dims
- MLP policy (77K-290K params), PPO with GAE
- Training data: ~32K hourly bars, validation: 70 days (1680 bars)

**Results**:

| Config | Val Return | Max DD | Sortino |
|--------|-----------|--------|---------|
| v1: 10M steps, 256h, lr=3e-4, ep=168 | 0.80x | -42% | -1.05 |
| v2: 50M steps, 512h, lr=1e-4, ep=336 | 0.88x | -36% | -0.60 |
| **Supervised baseline (ft30 selector)** | **42.5x** | **-1.1%** | **123.9** |

**Why it failed**:
1. **Information gap**: RL agent sees 10 basic features per bar vs supervised model's 96-step transformer with Chronos2 probabilistic forecasts, many engineered features
2. **No temporal memory**: MLP sees single bar; supervised model uses full 96-bar sequence context
3. **Sparse reward**: hourly equity changes provide weak learning signal for exploration
4. **Data scarcity**: ~32K training bars is tiny for RL's sample-hungry exploration
5. **Win rate plateau**: Both agents stuck at ~51% win rate, barely above random

**What could improve it**:
- Add Chronos2 forecast features to RL observation
- Use LSTM/GRU for temporal context
- Use supervised model predictions as features (distillation)
- Reward shaping with prediction-based auxiliary rewards
- Much more training data or data augmentation

**Conclusion**: The supervised approach's information advantage (Chronos2 forecasts, transformer sequence modeling) is decisive. Pure RL from price data alone can't match it. The C simulator is fast (365K steps/sec) but the feature gap dominates.

**Code**: `rl_trading/` directory
