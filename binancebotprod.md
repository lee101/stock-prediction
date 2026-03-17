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

**Latest local fresh-universe retest (March 16, 2026):**

The local workspace does not currently contain the legacy `mixed23_daily_{train,val}.bin`
artifacts or the documented `autoresearch_mixed23_daily/ent_anneal` checkpoint, and the
old 23-symbol stock+crypto universe is now clipped by stale symbols like `NET`, `SPY`, and
`QQQ`. A fresh 23-symbol mixed universe aligned through **2026-02-05** was rebuilt locally
and re-swept over the strongest regularized configs.

```
Fresh train data: pufferlib_market/data/mixed23_fresh_train.bin
Fresh val data:   pufferlib_market/data/mixed23_fresh_val.bin
Leaderboard:      pufferlib_market/autoresearch_mixed23_fresh_targeted_leaderboard.csv
Best checkpoint:  pufferlib_market/checkpoints/mixed23_fresh_targeted/reg_combo_2/best.pt
```

Targeted sweep result on the fresh universe:

| Config | Val Return | Val Sortino | Profitable% | Holdout Robust Score |
|--------|-----------:|------------:|------------:|---------------------:|
| `reg_combo_2` | **+15.72%** | **1.64** | **99.0%** | **-121.12** |
| `slip_10bps` | -12.59% | 0.26 | 0.0% | -173.50 |
| `clip_vloss` | -35.28% | -0.97 | 0.0% | -295.84 |
| `wd_01` | -56.61% | -1.04 | 0.0% | -349.80 |

Reproduction command:

```bash
python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/mixed23_fresh_train.bin \
    --val-data pufferlib_market/data/mixed23_fresh_val.bin \
    --time-budget 60 --max-trials 4 \
    --descriptions clip_vloss,wd_01,slip_10bps,reg_combo_2 \
    --leaderboard pufferlib_market/autoresearch_mixed23_fresh_targeted_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/mixed23_fresh_targeted \
    --periods-per-year 365 --max-steps-override 90 \
    --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
    --holdout-eval-steps 90 --holdout-n-windows 20 \
    --holdout-fee-rate 0.001 \
    --rank-metric holdout_robust_score
```

**Intraday replay stress test on the same fresh 90-day window (March 16, 2026):**

The leaderboard above is still based on the daily C environment, so each fresh
checkpoint was also replayed against aligned hourly bars from the same
`mixed23_fresh_val.bin` window (`2025-06-01` through **2026-02-05**) using
`python -m pufferlib_market.replay_eval --run-hourly-policy`.

Saved reports:

```
pufferlib_market/replay_eval_mixed23_fresh_reg_combo_2.json
pufferlib_market/replay_eval_mixed23_fresh_ent_anneal.json
pufferlib_market/replay_eval_mixed23_fresh_clip_vloss.json
```

Comparison:

| Config | Daily Return | Hourly Replay Return | Hourly Replay Sortino | Hourly Replay Max DD | Hourly Policy Return | Hourly Policy Orders |
|--------|-------------:|---------------------:|----------------------:|---------------------:|---------------------:|---------------------:|
| `reg_combo_2` | -22.66% | **-19.36%** | -1.32 | 67.22% | -74.11% | 966 |
| `ent_anneal` | **-18.78%** | -26.76% | **0.80** | **43.99%** | **-55.75%** | **446** |
| `clip_vloss` | -44.79% | -35.37% | 0.52 | 54.24% | -70.44% | 546 |

Interpretation:

- None of the previously saved fresh checkpoints are robust enough to deploy
  as-is once intraday execution stress is considered.
- Among the previously saved checkpoints, `reg_combo_2` remained the least bad
  on the more realistic "frozen daily action replayed hourly" path, which is
  the closest match to the current daily bot.
- `ent_anneal` is less bad than the others if the policy is naively re-run every
  hour, but that mode still thrashes badly and is not a valid deployment target.
- The next optimization target should be reducing intraday drawdown/churn under
  hourly replay, not just improving the daily close-to-close leaderboard.

Replay-ranked sweep actually run on the same four configs:

```bash
python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/mixed23_fresh_train.bin \
    --val-data pufferlib_market/data/mixed23_fresh_val.bin \
    --time-budget 60 --max-trials 4 \
    --descriptions ent_anneal,clip_vloss,wd_01,reg_combo_2 \
    --periods-per-year 365 --max-steps-override 90 \
    --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
    --holdout-eval-steps 90 --holdout-n-windows 20 \
    --holdout-fee-rate 0.001 \
    --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
    --replay-eval-hourly-root trainingdatahourly \
    --replay-eval-start-date 2025-06-01 \
    --replay-eval-end-date 2026-02-05 \
    --replay-eval-run-hourly-policy \
    --rank-metric replay_hourly_return_pct \
    --leaderboard pufferlib_market/autoresearch_mixed23_fresh_replay_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/mixed23_fresh_replay
```

Replay-ranked leaderboard result:

```
Leaderboard:     pufferlib_market/autoresearch_mixed23_fresh_replay_leaderboard.csv
Best checkpoint: pufferlib_market/checkpoints/mixed23_fresh_replay/ent_anneal/best.pt
```

| Config | Replay Hourly Return | Val Return | Holdout Robust Score |
|--------|---------------------:|-----------:|---------------------:|
| `ent_anneal` | **+57.22%** | -32.81% | -285.68 |
| `wd_01` | +5.56% | -36.78% | -282.17 |
| `clip_vloss` | -1.41% | -22.50% | -216.06 |
| `reg_combo_2` | -48.48% | **+22.54%** | **-148.00** |

Interpretation of the replay-ranked retrain:

- The ranking flips completely once the search is told to optimize the frozen
  daily action replay metric.
- A fresh `ent_anneal` retrain is now best on hourly replay, but it conflicts
  sharply with both the daily validation return and the multi-window holdout
  score, so it is not yet a clean production switch.
- `reg_combo_2` still dominates the old daily/holdout metrics, but it collapses
  under the replay target, which confirms the daily leaderboard alone is
  misleading for deployment.

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

## 60-Day Mixed23 Retest (2025-12-08 to 2026-02-05)

Re-exported the exact 60-day daily validation slice to keep replay_eval aligned with
the hourly replay window:

```bash
python -m pufferlib_market.export_data_daily \
  --symbols AAPL,NFLX,NVDA,ADBE,ADSK,COIN,GOOG,MSFT,PYPL,SAP,TSLA,BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,AAVEUSD,UNIUSD,DOTUSD,SHIBUSD,XRPUSD \
  --output /tmp/mixed23_val_60d_20251208_20260205.bin \
  --start-date 2025-12-08 --end-date 2026-02-05 --min-days 60
```

Saved 60-day comparison: `pufferlib_market/replay_eval_mixed23_60d_comparison.csv`

| Checkpoint | Daily Return | Daily MaxDD | Hourly Replay Return | Hourly Replay MaxDD | Hourly Policy Return |
|------------|--------------|-------------|----------------------|---------------------|----------------------|
| `ent_anneal` | `+62.37%` | `15.54%` | `+8.02%` | `34.77%` | `-65.56%` |
| `reg_combo_2` | `+30.24%` | `21.59%` | `+27.53%` | `34.21%` | `-47.06%` |
| `wd_01` | `+13.25%` | `20.19%` | `+11.11%` | `26.52%` | `-37.59%` |
| `clip_vloss` | `-23.73%` | `27.01%` | `-11.76%` | `22.01%` | `-51.20%` |

Takeaway:
- `ent_anneal` still wins on pure daily PnL and daily drawdown.
- `reg_combo_2` is the best recent checkpoint on frozen-daily hourly replay.
- `wd_01` is the most balanced recent compromise, but it is not a clear winner.
- None of these checkpoints are safe to deploy as true hourly policies; hourly-policy replay remains strongly negative for all four.
- Result: **no production switch yet**. Keep ranking on both daily and hourly-replay metrics until one checkpoint wins both with lower drawdown.

## 5bp Fill Buffer + Adaptive Meta Retest (2026-03-16)

The replay/holdout stack now requires daily bars to trade **through** a limit by
`5bp` before a fill, matching the hourly trader simulator's fill semantics more
closely. Relevant artifacts:

- `pufferlib_market/replay_eval_5bp_60d/*.json`
- `pufferlib_market/meta_replay_5bp_60d/*.json`
- `pufferlib_market/meta_replay_5bp_3window_sweep.csv`
- `pufferlib_market/mixed23_3window_strategy_summary.csv`

### Latest 60d window (2025-12-08 to 2026-02-05)

The best **current-window** adaptive selector was:

- selector: `sticky return`, `lookback=14d`, `recency_halflife=5d`, `switch_margin=0.01`
- file: `pufferlib_market/meta_replay_5bp_60d/sticky_return_lb14_hl5_sm001.json`

Current-window comparison:

| Strategy | Daily Return | Daily MaxDD | Hourly Replay Return | Hourly Replay MaxDD |
|----------|--------------|-------------|----------------------|---------------------|
| `ent_anneal` | `+62.37%` | `15.54%` | `+8.02%` | `34.77%` |
| `reg_combo_2` | `+30.24%` | `21.59%` | `+27.53%` | `34.21%` |
| `wd_01` | `+13.25%` | `20.19%` | `+11.11%` | `26.52%` |
| `meta sticky return 14/5 + sm=0.01` | `+74.26%` | `13.83%` | `+41.27%` | `24.86%` |

That selector only switched `3` times over the 59 decision days, which is why it
looked attractive for live use.

### 3x60d robustness check

Replayed the same meta selector unchanged on the two immediately previous 60-day windows:

| Window | `reg_combo_2` Hourly Replay | Meta Hourly Replay | `reg_combo_2` Daily | Meta Daily |
|--------|-----------------------------|--------------------|---------------------|------------|
| `2025-12-08..2026-02-05` | `+27.53%` | `+41.27%` | `+30.24%` | `+74.26%` |
| `2025-10-09..2025-12-07` | `+29.23%` | `-47.79%` | `+20.57%` | `-28.48%` |
| `2025-08-10..2025-10-08` | `+21.33%` | `-4.75%` | `+21.57%` | `-20.46%` |

Takeaway:

- The latest-window meta winner is **not robust** across adjacent 60-day windows.
- The 3-window sweep in `pufferlib_market/meta_replay_5bp_3window_sweep.csv` did **not** find a selector that stayed positive on hourly replay across all three windows.
- `reg_combo_2` remains the only checkpoint in this batch that stayed positive on hourly replay on all three tested windows.
- `ent_anneal` still dominates the latest window on pure daily PnL, but it fails badly on the older windows.
- Result: **still no production switch**. The selector logic needs a better regime gate or a broader checkpoint set before it is safe to deploy live.

## Robust Daily Variant Sweep (2026-03-16)

I then ran a targeted daily sweep centered on the only family that had held up
across the three replay windows: `reg_combo_2`. This required exposing
`--smoothness-penalty` in `pufferlib_market.train` and wiring
`drawdown_penalty`, `smooth_downside_penalty`, and `smoothness_penalty`
through `pufferlib_market.autoresearch_rl`.

Sweep command:

```bash
source .venv313/bin/activate
PYTHONPATH=$PWD/PufferLib:$PYTHONPATH python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_fresh_train.bin \
  --val-data pufferlib_market/data/mixed23_fresh_val.bin \
  --time-budget 180 --max-trials 8 \
  --descriptions robust_reg_wd02,robust_reg_tp005,robust_reg_tp01,robust_reg_tp005_sds02,robust_reg_tp005_dd002,robust_reg_tp005_sm001,robust_reg_tp005_ent,robust_reg_h512_tp005 \
  --periods-per-year 365 --max-steps-override 90 \
  --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
  --holdout-eval-steps 90 --holdout-n-windows 20 \
  --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5 \
  --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
  --replay-eval-hourly-root trainingdatahourly \
  --replay-eval-start-date 2025-06-01 \
  --replay-eval-end-date 2026-02-05 \
  --replay-eval-fill-buffer-bps 5 \
  --rank-metric replay_hourly_return_pct \
  --leaderboard pufferlib_market/autoresearch_mixed23_fresh_robust_leaderboard.csv \
  --checkpoint-root pufferlib_market/checkpoints/mixed23_fresh_robust
```

Artifacts:

- `pufferlib_market/autoresearch_mixed23_fresh_robust_leaderboard.csv`
- `pufferlib_market/checkpoints/mixed23_fresh_robust/*/best.pt`
- `pufferlib_market/mixed23_robust_3window_results.csv`
- `pufferlib_market/mixed23_robust_3window_summary.csv`

Fresh full-val leaderboard:

| Config | Replay Hourly Return | Val Return | Holdout Robust |
|--------|---------------------:|-----------:|---------------:|
| `robust_reg_tp005_sds02` | `+7.31%` | `-5.59%` | `-147.26` |
| `robust_reg_tp005_ent` | `-1.97%` | `-7.69%` | `-191.95` |
| `robust_reg_tp01` | `-3.57%` | `+64.43%` | `-95.92` |
| `robust_reg_h512_tp005` | `-18.92%` | `+49.42%` | `-71.64` |

Three-window replay retest:

| Strategy | Mean Hourly Return | Worst Hourly Return | Worst Hourly MaxDD |
|----------|--------------------:|--------------------:|-------------------:|
| `reg_combo_2` | `+26.03%` | `+21.33%` | `34.21%` |
| `robust_reg_tp01` | `+6.39%` | `-15.24%` | `30.97%` |
| `robust_reg_h512_tp005` | `+21.79%` | `-48.55%` | `53.87%` |
| `robust_reg_tp005_ent` | `-14.69%` | `-23.22%` | `41.80%` |
| `ent_anneal` | `-14.64%` | `-42.43%` | `49.40%` |
| `robust_reg_tp005_sds02` | `-23.33%` | `-44.64%` | `46.92%` |

Takeaway:

- No new daily checkpoint dominated `reg_combo_2` across the same three replay windows.
- `robust_reg_tp01` is the only new branch worth carrying forward: on the latest
  60-day window it lowers hourly replay max drawdown from `34.21%` to `25.80%`,
  but hourly replay return also drops from `+27.53%` to `+4.01%`, and it turns
  negative on the older window.
- Result: **still no production switch**. Keep `reg_combo_2` as the current
  robustness anchor and treat `robust_reg_tp01` as the main lower-drawdown branch
  for the next follow-up sweep.

## Key Learnings
1. **More diverse symbols = exponentially better**: 3-sym 0% positive → 8-sym 19% → 23-sym 46%
2. **Stocks + crypto together beats either alone**: Uncorrelated assets maximize RL opportunity
3. **23 is the sweet spot**: 32 symbols (+15% OOS) < 23 symbols (+27% OOS)
4. **Daily >> hourly**: Confirmed across all symbol counts
5. **5-min timeboxed training**: Prevents overfitting
6. **ent_anneal is champion**: entropy annealing 0.08→0.02 with anneal-LR
7. **trade_penalty counterproductive at 0% fee**: Was #1 at 10bps, now hurts
8. **Single-position model**: C env trains single-position. Don't try multi-position in Python backtest
9. **Adaptive checkpoint switching can look amazing on one window and fail on adjacent windows**: rank selectors on multiple recent windows, not just the latest slice
