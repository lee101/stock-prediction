# Neural Daily Production Trading Setup

## March 17, 2026 Active Deployment

- Sortino-first active checkpoint: `neuraldailytraining/checkpoints/active_latest.pt` -> `neuraldailytraining/checkpoints/neuraldaily_broad24_goodness_20260317_cap24x2_epoch001/epoch_0001.pt`
- Active deployment config: `neuraldailytraining/checkpoints/active_latest.json`
- Current promoted live settings: `account_fraction=0.2`, `risk_threshold=1.0`, `confidence_threshold=None`
- 24-symbol live universe: `EQIX GS COST CRM AXP BA GE LLY AVGO SPY SHOP GLD PLTR MCD V VTI QQQ MA SAP COUR ADBE INTC QUBT BTCUSD`
- Latest sweep reports:
  - return-heavy selection: `analysis/neural_daily_deploy_sweep_20260317_fast.json`
  - sortino-first reselection: `analysis/neural_daily_deploy_selection_20260317_sortino.json`
  - live-sized broad24 retune: `analysis/neural_daily_deploy_sweep_20260317_broad24_sizing.json`
  - live-sized active selection: `analysis/neural_daily_deploy_selection_20260317_broad24_sizing.json`

### Current Live Command
```bash
python neural_daily_trade_with_probe.py \
  --deployment-config neuraldailytraining/checkpoints/active_latest.json \
  --once
```

### Re-run the Promotion Sweep
```bash
python sweep_neural_daily_deployment.py \
  --checkpoint <candidate1.pt> <candidate2.pt> ... \
  --baseline-checkpoint neuraldailytraining/checkpoints/alldata_dec12/epoch_0001.pt \
  --symbols EQIX GS COST CRM AXP BA GE LLY AVGO SPY SHOP GLD PLTR MCD V VTI QQQ MA SAP COUR ADBE INTC QUBT BTCUSD \
  --days 60 \
  --window-count 2 \
  --window-stride-days 30 \
  --account-fractions 0.05,0.1,0.15,0.2 \
  --risk-thresholds 0.5,1.0 \
  --confidence-thresholds none,0.35,0.5 \
  --selection-metric sortino_p25 \
  --promote
```

## Overview
Enhanced neural daily trading system with **probe mode risk controls** and **leverage cost modeling**.

## What's New

### 1. Enhanced Simulator with Leverage Costs ✅
**File:** `neuraldailymarketsimulator/simulator.py`

**Features:**
- Tracks leverage at start of each day
- Charges 6.5% annual interest (0.0178% daily) on leveraged amounts
- Formula: If `positions_value > equity`, charge interest on `(positions_value - equity)`
- Reports: `leverage`, `leverage_cost`, `max_leverage`, `total_leverage_costs`

**Usage:**
```bash
python -m neuraldailymarketsimulator.simulator \
  --checkpoint neuraldailytraining/checkpoints/neuraldaily_20251118_015841/epoch_0006.pt \
  --days 10 \
  --data-root trainingdata/train \
  --start-date 2025-11-07 \
  --symbols EQIX GS COST CRM AXP BA GE LLY AVGO SPY SHOP GLD PLTR MCD V VTI QQQ MA SAP COUR ADBE INTC QUBT BTCUSD ETHUSD UNIUSD LINKUSD
```

**Output:**
```
Date                  Equity         Cash     Return   Leverage    LevCost
2025-11-07            0.9992       0.0000    -0.0008       0.00x   0.000000
2025-11-10            1.0174       0.0000     0.0182       1.00x   0.000000
...

Simulation Summary
Final Equity      : 1.0090
Net PnL           : 0.0090
Max Leverage      : 1.00x
Total Lev. Costs  : 0.000000
```

---

### 2. Production Trading with Probe Mode 🔬
**File:** `neural_daily_trade_with_probe.py`

**Probe Mode Features (ON by default):**

#### Global Probe Mode
- **Trigger:** Account has negative day PnL
- **Action:** All symbols trade with minimal notional ($300 max)
- **Duration:** Next trading day only
- **Integration:** Uses `src/risk_state.py` (same as trade_stock_e2e.py)

#### Per-Symbol Probe Mode
- **Trigger:** Last 2 trades for that symbol have negative total PnL%
- **Action:** Symbol trades with minimal notional ($300 max)
- **Duration:** Until 2 profitable trades occur
- **Storage:** `state/neural_trade_history_{suffix}.jsonl`

#### Normal Mode
- **Notional:** `account_equity * account_fraction * allocation`
- **Default:** 15% of account per symbol
- **Respects:** Model's risk threshold

**Key Benefits:**
1. **Protects capital** during losing streaks
2. **Automatic recovery** when performance improves
3. **Per-symbol tracking** isolates bad performers
4. **Global safety net** for account-wide losses

---

## Production Deployment Checklist

### Pre-Production Testing

1. **Run Latest Data Simulation**
```bash
# Test on most recent 10 days
python -m neuraldailymarketsimulator.simulator \
  --checkpoint neuraldailytraining/checkpoints/neuraldaily_20251118_015841/epoch_0006.pt \
  --days 10 \
  --data-root trainingdata/train \
  --start-date 2025-11-07 \
  --symbols EQIX GS COST CRM AXP BA GE LLY AVGO SPY SHOP GLD PLTR MCD V VTI QQQ MA SAP COUR ADBE INTC QUBT BTCUSD ETHUSD UNIUSD LINKUSD
```

**Results:** +0.9% over 10 days (Nov 7-16)

2. **Verify Probe Mode Works**
```bash
# Test with probe mode enabled (default)
python neural_daily_trade_with_probe.py \
  --checkpoint neuraldailytraining/checkpoints/neuraldaily_20251118_015841/epoch_0006.pt \
  --once \
  --symbols SPY QQQ
```

---

### Paper Trading (PAPER=1)

```bash
# Start with paper trading (PAPER=1 is default)
python neural_daily_trade_with_probe.py \
  --checkpoint neuraldailytraining/checkpoints/neuraldaily_20251118_015841/epoch_0006.pt \
  --symbols EQIX GS COST CRM AXP BA GE LLY AVGO SPY SHOP GLD PLTR MCD V VTI QQQ MA SAP COUR ADBE INTC QUBT BTCUSD ETHUSD UNIUSD LINKUSD \
  --account-fraction 0.15 \
  --interval 300
```

**Monitor:**
- Check logs: `logs/neural_daily_trade_with_probe.log`
- Watch for probe mode triggers: `🚨 GLOBAL PROBE MODE ACTIVE`
- Verify position sizes match expectations

---

### Gradual Ramp-Up to Live (PAPER=0)

#### Phase 1: Single Symbol Test
```bash
# Start with just SPY
PAPER=0 python neural_daily_trade_with_probe.py \
  --checkpoint neuraldailytraining/checkpoints/neuraldaily_20251118_015841/epoch_0006.pt \
  --symbols SPY \
  --account-fraction 0.05 \
  --interval 300
```

**Run for:** 2-3 days
**Monitor:** Daily PnL, probe triggers, fill rates

#### Phase 2: Add Core Symbols
```bash
# Add liquid stocks + BTC/ETH
PAPER=0 python neural_daily_trade_with_probe.py \
  --checkpoint neuraldailytraining/checkpoints/neuraldaily_20251118_015841/epoch_0006.pt \
  --symbols SPY QQQ BTCUSD ETHUSD \
  --account-fraction 0.10 \
  --interval 300
```

**Run for:** 1 week
**Monitor:** Performance across symbols, correlation of losses

#### Phase 3: Full Portfolio
```bash
# All trained symbols
PAPER=0 python neural_daily_trade_with_probe.py \
  --checkpoint neuraldailytraining/checkpoints/neuraldaily_20251118_015841/epoch_0006.pt \
  --symbols EQIX GS COST CRM AXP BA GE LLY AVGO SPY SHOP GLD PLTR MCD V VTI QQQ MA SAP COUR ADBE INTC QUBT BTCUSD ETHUSD UNIUSD LINKUSD \
  --account-fraction 0.15 \
  --interval 300
```

---

## Monitoring Commands

### Check Account Status
```bash
# Alpaca account
python alpaca_cli.py status

# Stock positions
python stock_cli.py status
```

### Check Probe State
```python
from src.risk_state import resolve_probe_state

probe_state = resolve_probe_state()
print(f"Force Probe: {probe_state.force_probe}")
print(f"Reason: {probe_state.reason}")
print(f"Probe Date: {probe_state.probe_date}")
```

### View Trade History
```python
from neural_daily_trade_with_probe import _get_recent_trade_pnl_pcts

# Check last 2 trades for SPY
pnls = _get_recent_trade_pnl_pcts("SPY", "buy", limit=2)
print(f"SPY recent PnLs: {pnls}")
```

---

## Risk Controls Summary

| Control | Trigger | Action | Duration |
|---------|---------|--------|----------|
| **Global Probe** | Negative day PnL | All symbols → $300 max notional | Next trading day |
| **Symbol Probe** | Last 2 trades negative | That symbol → $300 max notional | Until 2 wins |
| **Leverage Cost** | Positions > Equity | 6.5% annual interest | Ongoing |
| **Account Fraction** | Always | Max 15% per symbol | Always |
| **Risk Threshold** | Model config | Max 1.0x allocation | Always |

---

## Key Files

### Production
- `neural_daily_trade_with_probe.py` - Main trading script with probe mode
- `src/risk_state.py` - Global probe state management
- `state/neural_trade_history_{suffix}.jsonl` - Per-symbol trade history

### Simulation
- `neuraldailymarketsimulator/simulator.py` - Enhanced simulator with leverage costs

### Configuration
- Model: `neuraldailytraining/checkpoints/neuraldaily_20251118_015841/epoch_0006.pt`
- Symbols: 27 (from manifest.json)
- Risk Threshold: 1.0
- Maker Fee: 0.08%
- Leverage Rate: 6.5% annual

---

## Performance Notes

### Recent Simulation Results
- **Nov 1-10:** +28.5% (10 days)
- **Nov 7-16:** +0.9% (10 days)

**⚠️ High Variance Warning:**
The 30x difference in performance suggests:
1. Model may be overfitting to certain market conditions
2. More extensive backtesting needed across different periods
3. Gradual ramp-up is CRITICAL

### Recommendations
1. Start with paper trading for 1 week
2. Monitor probe mode triggers carefully
3. Track correlation between symbols
4. Set daily PnL alerts
5. Review every 2-3 days during ramp-up

---

## Emergency Stop

If things go wrong:
```bash
# Kill all watchers
pkill -f "neural_daily_trade_with_probe"

# Check positions
python stock_cli.py status

# Close all positions manually if needed
python alpaca_cli.py close-all
```

---

## Questions?

- Probe mode not triggering? Check `state/global_risk_state_{suffix}.jsonl`
- Trade history not saving? Check `state/neural_trade_history_{suffix}.jsonl`
- Leverage costs too high? Review position sizing in simulation first
- Model performance poor? Run longer backtests across multiple periods
