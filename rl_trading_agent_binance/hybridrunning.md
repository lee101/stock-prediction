# Binance RL+LLM Hybrid Trading System

## Overview
Combines a PufferLib PPO-trained RL agent with LLM reasoning (Gemini 3.1 Flash Lite + thinking HIGH) for spot crypto trading on Binance. Trades BTC, ETH, SOL, DOGE, SUI, AAVE.

## Architecture
1. **RL Model** (PufferLib PPO, h1024, 100M steps) generates base trading signals from MKTD features
2. **LLM Layer** (Gemini 3.1 Flash Lite, thinking=HIGH) reviews RL signals with market context and refines entry/exit
3. **Execution** routes orders to FDUSD pairs (BTC/ETH, zero fees) or USDT pairs (altcoins)

## Model
- **Best model**: `binance6_ppo_v1_h1024_100M` (38MB)
- **R2 location**: `s3://models/stock/models/binance6_ppo_v1_h1024_100M.pt`
- **Local**: `rl-trainingbinance/checkpoints/binance6_ppo_v1_h1024_100M/best.pt`
- **Config**: obs=107, actions=13, hidden=1024, MLP
- **Training**: 100M PPO steps, anneal-LR, ent-coef=0.05, 6 symbols

## OOS Backtest Results (Mar 2-9, 2026)

| Approach | Model | Return | Sortino | PnL ($10k) |
|----------|-------|--------|---------|------------|
| **Gemini 3.1 + thinking HIGH** | **100M** | **+6.82%** | **42.94** | **+$873** |
| Hybrid v1 (DeepSeek) | 100M | +5.81% | 47.66 | +$730 |
| Hybrid v4 (DeepSeek fresh) | fresh 100M | +3.74% | 24.36 | +$517 |
| Hybrid v2 (DeepSeek 300M) | 300M | +3.16% | 17.70 | +$449 |
| RL-only v4 | fresh 100M | +1.38% | 10.44 | +$201 |
| RL-only v1 | 100M | +0.45% | 8.29 | +$52 |
| RL-only v2 | 300M | -0.47% | -3.21 | -$24 |

**Key findings**:
- **Gemini 3.1 Flash Lite + thinking HIGH is the best**: +6.82% in 7 days, +17% over DeepSeek hybrid
- Hybrid consistently beats RL-only (6.82% vs 0.45%)
- 100M steps (less overfit) beats 300M OOS
- Max drawdown only 1.23% with Gemini (tight risk control)

## Pair Configuration

| Symbol | Binance Pair | Quote | Maker Fee | Max Position |
|--------|-------------|-------|-----------|-------------|
| BTCUSD | BTCFDUSD | FDUSD | 0.00% | 25% |
| ETHUSD | ETHFDUSD | FDUSD | 0.00% | 20% |
| SOLUSD | SOLUSDT | USDT | 0.10% | 15% |
| DOGEUSD | DOGEUSDT | USDT | 0.10% | 10% |
| SUIUSD | SUIUSDT | USDT | 0.10% | 15% |
| AAVEUSD | AAVEUSDT | USDT | 0.10% | 15% |

FDUSD pairs are Binance.com only (not Binance.US). System auto-falls back to USDT.

## Running

### Backtest (simulation)
```bash
# Hybrid backtest with Gemini thinking
python /tmp/run_binance_hybrid.py \
  --model gemini-3.1-flash-lite-preview \
  --thinking-level HIGH \
  --parallel 5 \
  --symbols BTCUSD ETHUSD SOLUSD \
  --checkpoint rl-trainingbinance/checkpoints/binance6_ppo_v1_h1024_100M/best.pt \
  --days 7

# RL-only comparison
python /tmp/run_binance_hybrid.py --rl-only --symbols BTCUSD ETHUSD SOLUSD --days 7

# Full comparison (RL + hybrid)
python /tmp/run_binance_hybrid.py --compare --symbols BTCUSD ETHUSD SOLUSD --days 7
```

### Dry Run (live data, no orders)
```bash
python rl_trading_agent_binance/trade_binance_live.py \
  --dry-run \
  --once \
  --model gemini-3.1-flash-lite-preview \
  --thinking-level HIGH \
  --symbols BTCUSD ETHUSD SOLUSD
```

### Production
```bash
python rl_trading_agent_binance/trade_binance_live.py \
  --live \
  --model gemini-3.1-flash-lite-preview \
  --thinking-level HIGH \
  --symbols BTCUSD ETHUSD SOLUSD DOGEUSD SUIUSD AAVEUSD \
  --interval 3600
```

## Stablecoin Management
The system automatically converts between FDUSD and USDT:
- BTC/ETH trades need FDUSD (zero fees on Binance.com)
- Altcoin trades need USDT
- Before placing an order, `ensure_quote_balance()` checks if conversion is needed
- Uses `binance_conversion.py` for FDUSD<->USDT swaps

## Files
- `run_hybrid.py` - Backtest engine with RL+LLM simulation
- `trade_binance_live.py` - Production trader with order execution
- `rl_trading_agent_binance_prompt.py` - Prompt builder for live trading
- `../llm_hourly_trader/providers.py` - Multi-provider LLM wrapper (Gemini, OpenAI, Anthropic, DeepSeek)
- `../rl-trainingbinance/train.sh` - RL training script
