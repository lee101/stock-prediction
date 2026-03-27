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
# Uses 6-model ensemble (tp10+s15+s36+gamma_995+muon_wd_005+h1024_a40 softmax_avg) by default
```

### Paper Snapshot (2026-03-27 17:55 UTC)
- **6-model ensemble** — NEW BEST, deployed to paper config (2026-03-27)
- @5bps: 0/111 neg, med=+58.0%, p10=+45.4%, worst=+36.6%

### Active 6-Model Ensemble (softmax_avg)
| Checkpoint | Config | Notes |
|-----------|--------|-------|
| `pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_10/best.pt` | tp10, ~6.9M steps | Conservative anchor |
| `pufferlib_market/checkpoints/stocks12_seed_sweep/tp05_s15/best.pt` | tp05, seed=15, 35M steps | Ensemble member |
| `pufferlib_market/checkpoints/stocks12_seed_sweep/tp05_s36/best.pt` | tp05, seed=36, 35M steps | Ensemble member |
| `pufferlib_market/checkpoints/stocks12_v2_sweep/stock_gamma_995/best.pt` | gamma=0.995, tp05, h=1024 | 4th member |
| `pufferlib_market/checkpoints/stocks12_v2_sweep/muon_wd_005/best.pt` | muon optimizer, wd=0.005 | 5th member |
| `pufferlib_market/checkpoints/stocks12_v2_sweep/h1024_a40/best.pt` | h=1024, trained on A40 | 6th member |

### Marketsim Status — EXHAUSTIVE EVAL (all 111 possible 90d windows, no early stop)
**2026-03-27 exhaustive eval** (stocks12_daily_val.bin, 111 windows = complete validation set):

**Production progression:**
| Ensemble | Med@5bps | P10@5bps | Worst@5bps | Neg/111 | Status |
|----------|----------|----------|------------|---------|--------|
| **6-model (tp10+s15+s36+gamma+muon+h1024)** | **+58.0%** | **+45.4%** | **+36.6%** | **0/111** | ✓ CURRENT |
| 5-model (+muon_wd_005) | +58.4% | +43.4% | +35.2% | 0/111 | best median |
| 4-model (+gamma_995) | +55.9% | +42.9% | +29.7% | 0/111 | superseded |
| 3-model (tp10+s15+s36) | +50.9% | +36.6% | +27.3% | 0/111 | superseded |
| 3-model (s123+s15+s36) | +46.3% | +28.6% | +21.0% | 0/111 | superseded |

**Slippage robustness (6-model):**
| Slippage | Med | P10 | Worst | Neg/111 |
|----------|-----|-----|-------|---------|
| 0bps | +54.5% | +42.6% | +30.6% | 0/111 |
| 5bps | +58.0% | +45.4% | +36.6% | 0/111 |
| 10bps | +61.5% | +45.5% | +36.6% | 0/111 |
| 20bps | +64.0% | +47.3% | +38.3% | 0/111 |

**Key v2_sweep candidates tested (exhaustive, as 5th/6th member of 4-model base):**
- `muon_wd_005` standalone 5th: med=58.4%, p10=43.4%, worst=35.2% ✓
- `h1024_a40` standalone 5th: med=55.9%, p10=44.1%, worst=34.1% (ties median)
- `stock_drawdown_pen` 5th: med=57.1%, p10=43.2%, worst=31.5% ✓
- `slip_5bps` 5th: med=54.9%, p10=44.4%, worst=32.8% (median drops)
- `stock_ent_05` standalone: +6.1%, 52/111 neg ✗ BAD

**CRITICAL**: Do NOT use `evaluate_holdout --seed 42` for deployment decisions.
Use exhaustive eval (all 111 windows). The `evaluate_holdout` formula selects different windows than canonical:
- s36 evaluate_holdout: -17.5%, 48/50 neg → WRONG (exhaustive: +27.9%, 1/111 neg)
- stock_ent_05 evaluate_holdout: +40.7%, 0/50 neg → WRONG (exhaustive: +6.1%, 52/111 neg)

### Eval Command (exhaustive ensemble eval)
```bash
source .venv313/bin/activate
python /tmp/test_tp10_comprehensive.py
# Or run scripts/eval_stocks12_seeds.py for individual model screening
```

### Config Details
- tp10: h=1024, trade_penalty=0.10, anneal_lr=True, ~6.9M steps (v2_sweep config)
- tp05_s15: seed=15, 128 envs, no bf16, 35M steps
- tp05_s36: seed=36, 128 envs, no bf16, 35M steps

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

### Goal — UPDATED
Found 6-model (tp10+s15+s36+gamma+muon_wd005+h1024_a40), med=58.0%, p10=45.4%, 0/111 neg. Deployed.
Full v2_sweep 5th-member search complete. muon_wd005 best 5th, h1024_a40 best 6th.
Next: explore whether further ensemble expansion is possible, or pivot to live deployment.

### Ensemble Discovery (2026-03-27)
- Approach: exhaustively eval all v2_sweep models as ensemble candidates
- **KEY FIND**: `stock_trade_pen_10` (tp=0.10, ~6.9M steps) replaces s123 in ensemble
  - tp10+s15+s36: med=50.9%, p10=36.6%, worst=27.3%, 0/111 neg — beats all prior ensembles
  - tp10 is a "conservative anchor" model (nearly do-nothing standalone, 5/111 neg, ~0% solo return)
  - Mechanism: high trade_penalty forces highly selective entries → ensemble diversity improves
- Seed sweep (baseline, tp=0.0, 10M steps): tested 300-611 (130+ seeds), best s411 (7/111 neg) — no qualifiers
- v2_sweep exhaustive: `stock_ent_05` is BAD (52/111 neg exhaustive despite 0/50 on evaluate_holdout)
- `evaluate_holdout --seed 42` is WRONG — uses different windows than canonical. Never use for deployment.

### Eval Pipeline
```bash
# Screen new seeds (50-window):
source .venv313/bin/activate && python scripts/eval_stocks12_seeds.py \
  --checkpoint-root pufferlib_market/checkpoints/stocks12_baseline5 \
  --data-path pufferlib_market/data/stocks12_daily_val.bin --n-windows 50

# Confirm with exhaustive (111-window):
python /tmp/test_current_prod_exhaustive.py
```

### Qualification Criteria (to beat 6-model ensemble)
1. Ensemble test: med > 58.0% AND p10 > 45.4% AND 0/111 neg (must beat ALL key metrics at 5bps)
2. Or: standalone exhaustive: ≤1/111 neg AND med > 27.9% (better than s36 for replacement)

### Priority: Disable Chronos2
The unified-stock-trader (Chronos2, LIVE) is negative on all periods 7d+. Consider stopping it and enabling daily-rl-trader for live trading once 90-day live performance confirmed on paper.

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
