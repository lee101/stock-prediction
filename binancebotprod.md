# Binance Crypto Trading Bot — Production Setup Guide

## Overview

Daily RL + LLM hybrid trading bot for Binance crypto. Trains on 8-14 symbols, executes on FDUSD (0% fee) and USDT (10bps fee) pairs.

## Architecture

```
Daily at UTC midnight:
  1. Refresh Chronos2 forecasts (h24 for each symbol)
  2. Run RL inference (daily trade_pen model, 8+ symbols)
  3. Build LLM prompt (Gemini 2.5 Flash) with:
     - RL signal + confidence
     - Chronos2 h24 forecast (predicted close/high/low)
     - Current position, P&L, hold days
     - Previous reasoning + decision history
     - Market regime (trend, SMA, volatility)
  4. LLM decides: allocation (-5x to +5x), buy/sell prices
  5. Place limit orders on Binance
  6. Monitor fills, trailing stop (0.3%), max hold (72h)
```

## Prerequisites

### Python Environment
```bash
# Use .venv313 (Python 3.13.9)
source .venv313/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -e toto/  # if needed

# Build C extension for RL
cd pufferlib_market && python setup.py build_ext --inplace && cd ..
```

### API Keys
```bash
# Required in env_real.py or environment:
export GEMINI_API_KEY="..."          # Gemini 2.5 Flash for LLM decisions
export BINANCE_API_KEY="..."         # Binance spot/margin trading
export BINANCE_SECRET="..."          # Binance API secret
```

### GPU Setup (after restart)
```bash
# Fix NVIDIA drivers if needed
sudo modprobe nvidia && sudo modprobe nvidia-uvm
nvidia-smi  # verify GPU visible
```

## Data Pipeline

### 1. Download Historical Data from Binance
```bash
# Hourly data (1h klines)
python scripts/collect_binance_vision_klines.py \
    --symbols BTCUSDT ETHUSDT SOLUSDT LTCUSDT AVAXUSDT DOGEUSDT LINKUSDT AAVEUSDT \
              SHIBUSDT DOTUSDT XRPUSDT NEARUSDT ICPUSDT APTUSDT \
    --interval 1h --start 2022-01-01 \
    --out-root trainingdatahourly/crypto

# Daily data (1d klines)
python scripts/collect_binance_vision_klines.py \
    --symbols BTCUSDT ETHUSDT SOLUSDT LTCUSDT AVAXUSDT DOGEUSDT LINKUSDT AAVEUSDT \
              SHIBUSDT DOTUSDT XRPUSDT NEARUSDT ICPUSDT APTUSDT \
    --interval 1d --start 2022-01-01 \
    --out-root trainingdata/train
```

Note: Rename USDT-suffixed CSVs to USD for consistency:
```bash
for f in trainingdatahourly/crypto/*USDT.csv; do
    mv "$f" "${f/USDT.csv/USD.csv}"
done
```

### 2. Export Binary Data for C Environment
```bash
# Daily (best for RL training)
python -m pufferlib_market.export_data_daily \
    --symbols BTCUSD,ETHUSD,SOLUSD,LTCUSD,LINKUSD,UNIUSD,DOGEUSD,AAVEUSD \
    --data-root trainingdata/train \
    --output pufferlib_market/data/crypto8_daily_train.bin \
    --end-date 2025-06-01

python -m pufferlib_market.export_data_daily \
    --symbols BTCUSD,ETHUSD,SOLUSD,LTCUSD,LINKUSD,UNIUSD,DOGEUSD,AAVEUSD \
    --data-root trainingdata/train \
    --output pufferlib_market/data/crypto8_daily_val.bin \
    --start-date 2025-06-01

# Hourly (price-only, no Chronos2 dependency)
python -m pufferlib_market.export_data_hourly_priceonly \
    --symbols BTCUSD,ETHUSD,SOLUSD,LTCUSD,LINKUSD,UNIUSD,DOGEUSD,AAVEUSD \
    --data-root trainingdatahourly \
    --output pufferlib_market/data/crypto8_hourly_train.bin \
    --end-date 2025-06-01
```

### 3. Chronos2 Forecast Training (LoRA Fine-tuning)

Train per-symbol LoRA adapters for Chronos2:
```bash
# Per-symbol LoRA training (daily frequency)
python -m strategytrainingneural.train_chronos2_lora \
    --symbol BTCUSD \
    --data-dir trainingdata/train \
    --output-dir models/chronos2_lora/BTCUSD \
    --context-length 512 \
    --epochs 10

# For all symbols:
for sym in BTCUSD ETHUSD SOLUSD LTCUSD LINKUSD DOGEUSD AAVEUSD AVAXUSD; do
    python -m strategytrainingneural.train_chronos2_lora \
        --symbol $sym --data-dir trainingdata/train \
        --output-dir models/chronos2_lora/$sym \
        --context-length 512 --epochs 10
done
```

Upload to R2 models bucket:
```bash
# Upload trained LoRA adapters
for sym in BTCUSD ETHUSD SOLUSD LTCUSD LINKUSD DOGEUSD AAVEUSD AVAXUSD; do
    aws s3 sync models/chronos2_lora/$sym s3://models/chronos2_lora/$sym --endpoint-url $R2_ENDPOINT
done
```

### 4. Refresh Forecast Cache
```bash
# Generate fresh Chronos2 h24 forecasts
python strategytrainingneural/collect_forecasts.py \
    --data-dir trainingdata/train \
    --cache-dir alpacanewccrosslearning/forecast_cache/crypto_daily \
    --context-length 512 --batch-size 64 \
    --symbol BTCUSD --symbol ETHUSD --symbol SOLUSD \
    --symbol LTCUSD --symbol LINKUSD --symbol DOGEUSD \
    --symbol AAVEUSD --symbol AVAXUSD \
    --frequency daily
```

## RL Training

### Autoresearch Sweep (find best hyperparameters)
```bash
# Daily sweep (recommended — daily RL wins 3.3x over hourly)
python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/crypto8_daily_train.bin \
    --val-data pufferlib_market/data/crypto8_daily_val.bin \
    --time-budget 300 --max-trials 40 \
    --periods-per-year 365 --max-steps-override 90 \
    --fee-rate-override 0.0 \
    --leaderboard pufferlib_market/autoresearch_crypto8_daily_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_crypto8_daily
```

### Current Best Configs

**PRODUCTION CHAMPION: 23-sym mixed daily ent_anneal**

```
Checkpoint: pufferlib_market/checkpoints/autoresearch_mixed23_daily/ent_anneal/best.pt
Symbols: 15 stocks + 8 crypto = 23 total
Data: pufferlib_market/data/mixed23_daily_{train,val}.bin
```

| Metric | Value |
|--------|-------|
| OOS Return (90d mean) | **+26.69%** |
| Annualized | **+160.1%** |
| Sortino | **2.21** |
| Profitable | **100%** (500/500 episodes) |
| Worst episode | +11.74% |
| Win rate | 57.8% |
| Trades per 90d | 62.4 |

**Evaluation command (authoritative C env):**
```bash
python -u -m pufferlib_market.evaluate \
    --checkpoint pufferlib_market/checkpoints/autoresearch_mixed23_daily/ent_anneal/best.pt \
    --data-path pufferlib_market/data/mixed23_daily_val.bin \
    --deterministic --hidden-size 1024 \
    --max-steps 90 --num-episodes 500 --seed 42 \
    --fill-slippage-bps 5 --periods-per-year 365
```

**Other strong configs (8-symbol crypto daily, fee=0):**
| Config | OOS Return | Sortino | Profitable% |
|--------|-----------|---------|-------------|
| clip_anneal | +18.84% | 1.85 | 100% |
| slip_10bps | +14.35% | 1.71 | 100% |
| wd_05 | +8.73% | 1.40 | 100% |

### Train Specific Config
```bash
# Best config: clip_anneal
python -u -m pufferlib_market.train \
    --data-path pufferlib_market/data/crypto8_daily_train.bin \
    --total-timesteps 999999999 --max-steps 90 \
    --hidden-size 1024 --lr 3e-4 --ent-coef 0.05 \
    --anneal-lr --anneal-clip --clip-eps 0.2 --clip-eps-end 0.05 \
    --fee-rate 0.0 --periods-per-year 365 \
    --checkpoint-dir pufferlib_market/checkpoints/prod_daily
```

### Evaluate Checkpoint
```bash
python -u -m pufferlib_market.evaluate \
    --checkpoint pufferlib_market/checkpoints/autoresearch_crypto8_daily/clip_anneal/best.pt \
    --data-path pufferlib_market/data/crypto8_daily_val.bin \
    --deterministic --hidden-size 1024 \
    --max-steps 90 --num-episodes 500 --seed 42 \
    --fill-slippage-bps 3 --periods-per-year 365
```

## LLM Daily Trading Backtest
```bash
# Backtest with Gemini LLM
python -u backtest_daily_llm.py \
    --symbols BTCUSD,ETHUSD,SOLUSD \
    --start-date 2025-10-01 --end-date 2025-12-31 \
    --model gemini-2.5-flash --fee-tier fdusd \
    --slippage-bps 3

# Buy-and-hold benchmark
python -u backtest_daily_llm.py \
    --symbols BTCUSD,ETHUSD,SOLUSD \
    --start-date 2025-10-01 --end-date 2025-12-31 \
    --no-llm --fee-tier fdusd
```

## Leverage Sweep
```bash
python -u pufferlib_market/sweep_leverage.py \
    --train-data pufferlib_market/data/crypto8_daily_train.bin \
    --val-data pufferlib_market/data/crypto8_daily_val.bin \
    --timeframe daily
```

## Production Deployment

### Binance Fee Tiers
| Quote Asset | Maker Fee | Pairs | Strategy |
|-------------|-----------|-------|----------|
| FDUSD | 0% | BTC, ETH, SOL, BNB | Primary — zero cost |
| USDT | 0.1% | Everything else | Secondary — only trade with >20bps edge |

### Risk Controls
| Control | Setting | Rationale |
|---------|---------|-----------|
| Max position per symbol | 20% of account | No single-symbol blow-up |
| Max total leverage | 3x (even if 5x available) | Leave margin buffer |
| Trailing stop | 0.3% from peak | Proven in hourly prod |
| Daily loss limit | -3% of account | Stop trading for the day |
| Max hold | 72 hours (3 days) | Prevent multi-day underwater |
| SMA filter | Price > SMA-24 for longs | Suppress trades in downtrends |
| Execution | Limit orders ONLY | Capture maker fee tier |

### Key Files
| File | Purpose |
|------|---------|
| `pufferlib_market/train.py` | PPO training for C env |
| `pufferlib_market/autoresearch_rl.py` | Hyperparameter sweep |
| `pufferlib_market/evaluate.py` | OOS evaluation |
| `pufferlib_market/export_data_daily.py` | Daily binary export |
| `pufferlib_market/export_data_hourly_priceonly.py` | Hourly export (no Chronos2) |
| `pufferlib_market/sweep_leverage.py` | Leverage + reward shaping sweep |
| `backtest_daily_llm.py` | LLM daily backtest |
| `trade_daily_rl.py` | Daily RL trading bot |
| `marketsimulator.py` | Python market simulator |
| `binanceprogress6.md` | Experiment log |

### Missing Models to Train
- [ ] Chronos2 LoRA adapters for: LINKUSD, UNIUSD, DOGEUSD, SHIBUSD, DOTUSD, XRPUSD
- [ ] Daily RL checkpoint for 11+ symbol dataset
- [ ] Chronos2 3-day forecast models (extend prediction_length)
- [ ] Cross-learning models (BTC+ETH+SOL joint prediction)

### Model Storage
```bash
# R2 bucket structure
models/
├── chronos2_lora/
│   ├── BTCUSD/          # Per-symbol LoRA adapters
│   ├── ETHUSD/
│   └── ...
├── pufferlib_checkpoints/
│   ├── crypto8_daily_clip_anneal.pt    # Best daily RL
│   ├── crypto8_daily_trade_pen_05.pt   # Alternative
│   └── ...
└── compiled_models/
    └── chronos2_torch_inductor/        # Compiled for fast inference
```

## Production Architecture

The C env trains a **single-position agent** — it picks ONE symbol at a time to be long/short on, or stays flat. This IS work-stealing: the RL evaluates all 23 symbols and picks the best opportunity each day.

```
Daily at UTC midnight:
  1. Load latest daily bars for all 23 symbols
  2. Export to MKTD binary (or compute features in Python)
  3. Run RL inference → picks ONE symbol + direction
  4. [Optional] Pass to Gemini for entry/exit price refinement
  5. Execute: close old position, open new one
  6. Monitor with trailing stop (0.3%) throughout the day
```

**Important**: The Python backtest (trade_mixed_daily.py) has feature computation
mismatches with the C env binary data. Always use C env evaluation (evaluate.py)
as the authoritative benchmark. The Python scripts are for live inference only,
where features must be computed on-the-fly from fresh market data.

## Key Learnings
1. **More diverse symbols = exponentially better**: 3-sym 0% positive → 8-sym 19% → 23-sym 46%
2. **Stocks + crypto together beats either alone**: Uncorrelated assets maximize RL opportunity
3. **23 is the sweet spot**: 32 symbols (+15% OOS) < 23 symbols (+27% OOS)
4. **Daily >> hourly**: Confirmed across all symbol counts
5. **5-min timeboxed training**: Prevents overfitting
6. **ent_anneal is champion**: entropy annealing 0.08→0.02 with anneal-LR
7. **trade_penalty counterproductive at 0% fee**: Was #1 at 10bps, now hurts
8. **Single-position model**: C env trains single-position. Don't try multi-position in Python backtest
