# Alpaca Production Trading Systems

## Production bookkeeping
- This file tracks only Alpaca-related live and paper systems.
- Keep updated with what is live, how it is launched, and the latest timestamped results.
- See `prod.md` for Binance systems (binance-hybrid-spot, binance-worksteal-daily).

---

## Current Live Snapshot (2026-03-26 04:22 UTC)
- **LIVE account**: supervisor `unified-stock-trader` is active; equity **$41,315.42**, cash **$10,298.17**, buying power **$20,596.34**, last_equity **$41,077.99**
- **LIVE positions/orders**: no stock positions open; only dust in `AVAXUSD`, `BTCUSD`, `ETHUSD`, `LTCUSD`, `SOLUSD`. Open orders: 3 `ETH/USD` GTC buy limits.
- **PAPER account**: `daily-rl-trader.service` installed but currently `inactive (dead)`; paper equity **$55,268.15**, last_equity **$54,078.69**, day change **+$1,189.46 (+2.20%)**, total unrealized **+$3,269.46**

---

## System 1: Alpaca Stock Trader (`unified-stock-trader`) — LIVE (supervisor)

### Identity
- **Bot**: `unified_hourly_experiment/trade_unified_hourly_meta.py`
- **Service manager**: supervisor program `unified-stock-trader`
- **Installed config**: `/etc/supervisor/conf.d/unified-stock-trader.conf`
- **Architecture**: Chronos2 hourly, multiple models + meta-selector

### Launch Command
```bash
.venv313/bin/python -u /nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/trade_unified_hourly_meta.py \
  --strategy wd06=/nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/checkpoints/wd_0.06_s42:8 \
  --strategy wd06b=/nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/checkpoints/wd_0.06_s1337:8 \
  --stock-symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV \
  --min-edge 0.001 --fee-rate 0.001 --max-positions 5 --max-hold-hours 5 \
  --trade-amount-scale 100.0 --min-buy-amount 2.0 \
  --entry-intensity-power 1.0 --entry-min-intensity-fraction 0.0 \
  --long-intensity-multiplier 1.0 --short-intensity-multiplier 1.5 \
  --meta-metric p10 --meta-lookback-days 14 --meta-selection-mode sticky \
  --meta-switch-margin 0.005 --meta-min-score-gap 0.0 --meta-recency-halflife-days 0.0 \
  --meta-history-days 120 --sit-out-if-negative --sit-out-threshold -0.001 \
  --market-order-entry --bar-margin 0.0005 --entry-order-ttl-hours 6 \
  --margin-rate 0.0625 --live --loop
```

### Environment
- `PYTHONPATH=/nvme0n1-disk/code/stock-prediction`
- `PYTHONUNBUFFERED=1`
- `CHRONOS2_FREQUENCY=hourly`
- `PAPER=0`

### Symbols
NVDA, PLTR, GOOG, DBX, TRIP, MTCH, NYT, AAPL, MSFT, META, TSLA, NET, BKNG, EBAY, EXPE, ITUB, BTG, ABEV (18 symbols)

### Active Models (2-strategy meta-selector)
| Strategy | Checkpoint | Epoch |
|----------|-----------|-------|
| wd06 | `unified_hourly_experiment/checkpoints/wd_0.06_s42` | best = epoch_008.pt |
| wd06b | `unified_hourly_experiment/checkpoints/wd_0.06_s1337` | best = epoch_008.pt |

### Marketsim Status (2026-03-27 fresh backtest, 90d validation)

**backtest_portfolio.py with all epochs, 17 production symbols:**

| Model | 1d | 7d | 30d | 90d | 120d | Best epoch |
|-------|----|----|-----|-----|------|-----------|
| wd_0.06_s42 | **+0.4%** (ep5) | -0.6% | -5.5% | — | -5.1% | ep5 for 1d only |
| wd_0.06_s1337 | +0.1% (ep20) | -0.4% | -3.0% | — | -4.8% | ep20 for 1d only |

**⚠️ CRITICAL: Both Chronos2 models are NEGATIVE on ALL periods 7d+.** Sortino < -5 on 30d+.
- Previous marketsim (2026-03-26): min_sortino=-11.15, mean_sortino=-6.28, min_return=-18.09%
- Root cause: market conditions (stocks/crypto selloff Q1 2026) diverged from training distribution
- **Action needed**: Retrain or replace with stocks12 PPO daily trader

**Execution parity caveat**: trade-log replay on TSLA+ABEV slice: exact_row_ratio=0.5 → sim PnL not fully production-equivalent

### Eval Command (Chronos2 hourly backtest)
```bash
source .venv313/bin/activate
python unified_hourly_experiment/backtest_portfolio.py \
  --checkpoint unified_hourly_experiment/checkpoints/wd_0.06_s42 \
  --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV \
  --validation-days 90 --sweep-epochs
```

### Monitoring
```bash
sudo supervisorctl status unified-stock-trader
sudo tail -50 /var/log/supervisor/unified-stock-trader.log
```

### Recent Incidents
- **2026-03-24 crash-loop**: Supervisor referenced 5 missing checkpoints (wd_0.04, wd_0.05_s42, wd_0.08_s42, wd_0.03_s42, stock_sortino_robust variants). Fixed by reducing to 2 strategies (wd_0.06_s42 + wd_0.06_s1337).
- **2026-03-25 3 bugs fixed** in `trade_unified_hourly.py`:
  1. `pending_close_retry`: positions stuck in pending_close with no exit order now retry force_close
  2. Cancel race condition: sleep(0.75) after cancel before new order (prevents "qty held for orders")
  3. Crypto qty: abs(qty)<1 wrongly treated fractional crypto as closed — fixed with notional check
- **ABEV incident (2026-03-25)**: $12k ABEV position (entered 2026-03-20 @ $2.73) had no exit order since 2026-03-24 due to race condition. Force_close fired 01:42 UTC, filled 2026-03-25 13:30 UTC @ $2.83 (~+3.7%).

---

## System 2: Alpaca Daily PPO Trader (`trade_daily_stock_prod.py`) — INSTALLED, INACTIVE (paper systemd)

### Identity
- **Service manager**: systemd unit `daily-rl-trader.service`
- **Installed unit**: `/etc/systemd/system/daily-rl-trader.service`
- **Architecture**: h=1024 MLP PPO, stocks12 (AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, JPM, V, AMZN, PLTR)

### Launch Command
```bash
# Paper (current service config):
.venv313/bin/python -u trade_daily_stock_prod.py --daemon --paper --allocation-pct 25

# Live:
source .venv313/bin/activate
python trade_daily_stock_prod.py --live
# Uses 3-model ensemble (s123+s15+s36 softmax_avg) by default
```

### Paper Snapshot (2026-03-27 11:20 UTC)
- **REVERTED** to s123+s15+s36 ensemble (stock_ent_05 was bad — see eval notes below)
- Previous equity 2026-03-25: **$55,268.15**, cash **$2,235.60**

### Active 3-Model Ensemble (softmax_avg) — RESTORED 2026-03-27
| Checkpoint | Config | Notes |
|-----------|--------|-------|
| `pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt` | tp05, seed=123, ~1.44M steps | Primary |
| `pufferlib_market/checkpoints/stocks12_seed_sweep/tp05_s15/best.pt` | tp05, seed=15, 35M steps | Ensemble member |
| `pufferlib_market/checkpoints/stocks12_seed_sweep/tp05_s36/best.pt` | tp05, seed=36, 35M steps | Ensemble member |

### Marketsim Status — EXHAUSTIVE EVAL (all 111 possible 90d windows, 5bps, no early stop)
**2026-03-27 exhaustive eval** (stocks12_daily_val.bin, 111 windows = complete validation set):

| Model | Med | P10 | Neg/111 | Notes |
|-------|-----|-----|---------|-------|
| **s123+s15+s36 ensemble** | **+46.3%** | **+28.6%** | **0/111** | ✓ PRODUCTION |
| s15 standalone | +30.0% | +15.8% | 0/111 | |
| s36 standalone | +27.9% | +10.1% | 1/111 | NOT collapsed! |
| s123 standalone | +16.8% | +10.5% | 0/111 | |
| stock_ent_05 standalone | +6.1% | -18.1% | 52/111 | ✗ BAD — evaluated_holdout was misleading |

**CRITICAL**: Do NOT use `evaluate_holdout --seed 42` for deployment decisions.
Use exhaustive eval (all 111 windows) or canonical `default_rng(42)` 50-window eval.
The `evaluate_holdout` formula selects different windows than canonical and gave wrong results:
- s36 evaluate_holdout: -17.5%, 48/50 neg → WRONG (exhaustive: +27.9%, 1/111 neg)
- stock_ent_05 evaluate_holdout: +40.7%, 0/50 neg → WRONG (exhaustive: +6.1%, 52/111 neg)
| tp05_s36 standalone | -17.5% | -26.1% | -4.0% | -32.3% | 48/50 | ✗ COLLAPSED |

### Eval Command (new ensemble)
```bash
source .venv313/bin/activate
# Individual model eval (use /tmp/eval_new_ensemble.py for full ensemble):
python -m pufferlib_market.evaluate_holdout \
  --checkpoint pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt \
  --data-path pufferlib_market/data/stocks12_daily_val.bin \
  --eval-hours 90 --n-windows 50 --seed 42 \
  --fee-rate 0.001 --fill-buffer-bps 5.0 --deterministic --no-early-stop
```

### Config Details
- s123: h=1024, lr=3e-4, ent=0.05, trade_penalty=0.05, dp=0.0, anneal_lr=True, seed=123, ~1.44M steps
- tp05_s15: seed=15, 128 envs, no bf16, 35M steps
- stock_ent_05: baseline (tp=0.0, ent=0.05, h=1024, anneal_lr=True), seed=42, 7.47M steps

### WARNING
Symbol conflict with unified-stock-trader (both trade AAPL/MSFT/NVDA/GOOG/META/TSLA/PLTR). Coordinate before enabling both simultaneously.

### Enable for Live
```bash
sudo systemctl start daily-rl-trader.service  # starts paper
# Edit service file to remove --paper flag for live trading
```

---

---

## Active Improvement Experiments (2026-03-27)

### Goal
The Chronos2 hourly models are failing (negative on 30d+ recent data). The stocks12 daily PPO ensemble is working (0/50 neg, med=53%). Priority: improve the daily PPO and explore better architectures.

### Running Experiments (local GPU, ~15-20 min each)

**Seed sweep (tp05, h=1024, anneal_lr, no bf16, 128 envs, 15M steps):**
- Seeds 51-59: `pufferlib_market/checkpoints/stocks12_new_seeds/tp05_s{N}/best.pt`
- Also: s42_35M (35M steps for comparison)

**Architecture sweep (tp05, seed=42, 15M steps):**
- h2048: `checkpoints/stocks12_arch_experiments/tp05_h2048`
- h512: `checkpoints/stocks12_arch_experiments/tp05_h512`
- stocks15: `checkpoints/stocks12_arch_experiments/tp05_stocks15_tp05` (15 symbols)
- stocks20: `checkpoints/stocks12_arch_experiments/tp05_stocks20` (20 symbols)

**Eval command (once training done):**
```bash
source .venv313/bin/activate
python scripts/eval_stocks12_seeds.py \
  --checkpoint-root pufferlib_market/checkpoints/stocks12_new_seeds \
  --data-path pufferlib_market/data/stocks12_daily_val.bin
```

### Next Steps
1. Evaluate all seeds/architectures with 50-window holdout
2. Any 0/50-neg seeds → add to ensemble
3. Best architecture → 100M step training on RunPod
4. Enable stocks12 daily PPO as primary Alpaca trader (disable Chronos2 or reduce weight)

---

## JAX Retrain Status (2026-03-26)
- JAX/Flax port: `binanceneural/jax_policy.py`, `binanceneural/jax_losses.py`, `binanceneural/jax_trainer.py`, `unified_hourly_experiment/train_jax_classic.py`
- Local parity tests pass; smoke train: `unified_hourly_experiment/checkpoints/alpaca_progress7_jax_smoke_20260326/epoch_001.flax`
- RunPod RTX 4090 bootstrap in progress for detached retrain; see `alpacaprogress7.md`

---

## Marketsim Realism Checklist (Alpaca Stocks)
- [x] Fee: 10bps maker (0.001)
- [x] Margin interest: 6.25% annual
- [x] Fill buffer: 5bps
- [x] Max hold: 5h forced exit (Chronos2) / 90-day windows (daily PPO)
- [x] Decision lag: >=1 bar
- [x] Binary fills for validation (not soft sigmoid)
- [x] Slippage test: 0, 5, 10, 20, 50 bps (daily PPO validated)
- [x] No EOD close for Chronos2; GTC orders hold overnight
- [ ] Execution parity for Chronos2: trade-log replay exact_row_ratio < 1.0 (work in progress)

---

## Key Findings / Guard Rails
- **NYT is poison**: was classified SHORT but rallied +51.6%. Still included for diversification but watch carefully.
- **Hold=5 critical** for Chronos2: hold=4 breaks 120d+, hold=6 breaks 7d.
- **Seed variance extreme** in PPO: seed=123 gives +16.5% median; seed=7 gives -13.6%.
- **Only trust 50-window EnsembleTrader with default_rng(42)** for deployment decisions (20-window seed=1337 is unreliable).
- **soft sigmoid fills have lookahead bias** — never trust training Sortino alone.
- **Lookahead in neuraldailytraining**: lag=0 binary shows +63.5%, lag=1 shows -45.6%. DO NOT DEPLOY muon_baseline models.
