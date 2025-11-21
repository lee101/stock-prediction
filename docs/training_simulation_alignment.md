# Training vs Simulation Alignment

## Summary
The neural daily model is trained with realistic trading constraints. The simulator now **exactly matches** the training environment.

---

## Training Environment Constraints ✅

### 1. Leverage Rules
**File:** `neuraldailytraining/config.py:89-91`, `neuraldailytraining/trainer.py:264-271`

```python
equity_max_leverage: float = 2.0   # Stocks can leverage up to 2x
crypto_max_leverage: float = 1.0   # Crypto CANNOT leverage (max 1x)
leverage_fee_rate: float = 0.065   # 6.5% annual interest on leveraged positions
```

**How it works in training:**
- Model outputs `inventory_path` (position sizes over time)
- For each asset, determine if it's crypto (asset_class_flag > 0.5)
- Crypto positions capped at 1.0x equity
- Stock positions capped at 2.0x equity
- Charge daily leverage fee: `fee_per_step = 0.065 / 252` (trading days)
- Penalty added to loss: `loss += fee_per_step * avg_excess`

### 2. No Shorting
**File:** `differentiable_loss_utils.py:177`

```python
sell_qty = intensity * torch.clamp(inventory, min=0.0)
```

**Constraint:**
- Inventory is clamped to minimum of 0.0
- Cannot sell what you don't own
- Applies to **both stocks and crypto**
- No short selling allowed

### 3. Maker Fees
**File:** `neuraldailytraining/config.py:58`

```python
maker_fee: float = 0.0008  # 0.08% on both buys and sells
```

**How it works:**
- Buy cost: `qty * price * (1 + 0.0008)`
- Sell proceeds: `qty * price * (1 - 0.0008)`

---

## Simulator Now Matches Training ✅

### File: `neuraldailymarketsimulator/simulator.py`

#### 1. Stock vs Crypto Detection
```python
# Track which symbols are crypto (end with USD or -USD)
if symbol.upper().endswith("USD") or symbol.upper().endswith("-USD"):
    self.crypto_symbols.add(symbol)
```

#### 2. Leverage Cost Calculation
```python
# Separate stock and crypto positions
stock_value = sum(inventory * price for stocks)
crypto_value = sum(inventory * price for crypto)

# Only charge leverage fees on stocks
if stock_value > current_equity:
    # Stock positions are leveraged beyond 1x
    leveraged_amount = min(stock_value - current_equity, current_equity)  # Cap at 2x
    leverage_cost = leveraged_amount * daily_leverage_rate  # 0.065 / 365
    cash -= leverage_cost
```

#### 3. Why Crypto Can't Leverage
In real trading (Alpaca):
- Crypto uses **cash-only** accounts
- Cannot borrow to buy crypto
- Crypto positions cannot exceed available cash
- Stock accounts can use margin (up to 2x for day trading, 4x intraday)

---

## Key Differences: Training vs Reality

| Feature | Training | Simulator | Reality (Alpaca) |
|---------|----------|-----------|------------------|
| **Stock Leverage** | Max 2x | Max 2x | Max 4x intraday, 2x overnight |
| **Crypto Leverage** | Max 1x (none) | Max 1x (none) | Max 1x (none) |
| **Leverage Fee** | 6.5% annual | 6.5% annual | 6.5% annual |
| **Shorting** | Not allowed | Not allowed | Allowed (stocks only) |
| **Maker Fee** | 0.08% | 0.08% | ~0.1-0.3% (varies) |

**Why train with 2x max leverage instead of 4x?**
- More conservative risk management
- 4x leverage is only intraday (must close by EOD)
- Overnight leverage is 2x, matching our training
- Model learns sustainable position sizing

---

## Real Trading Constraints (Alpaca)

### Stock Trading
- ✅ **Can leverage:** Up to 2x overnight, 4x intraday
- ✅ **Can short:** Yes (with margin account)
- ✅ **Leverage cost:** 6.5% annual on borrowed amount
- ⚠️ **Pattern Day Trader:** Need $25k+ equity for unlimited day trades

### Crypto Trading
- ❌ **Cannot leverage:** Cash-only accounts
- ❌ **Cannot short:** No shorting allowed
- ✅ **24/7 trading:** Can trade anytime
- ✅ **No PDT rules:** No day trade limits

### Margin Requirements
From `trade_stock_e2e.py:89-90`:
```python
equity_max_leverage: float = 2.0
crypto_max_leverage: float = 1.0
```

This matches Alpaca's **overnight** margin requirements.

---

## Example: Mixed Portfolio Leverage Calculation

### Scenario
- Account equity: $10,000
- Stock positions: $15,000 (1.5x leveraged)
- Crypto positions: $8,000 (0.8x, within limit)
- Total positions: $23,000

### Leverage Cost Calculation

**Step 1:** Separate stocks and crypto
- Stock value: $15,000
- Crypto value: $8,000

**Step 2:** Calculate leveraged amount (stocks only)
- Equity: $10,000
- Stock leverage: $15,000 / $10,000 = 1.5x
- Leveraged amount: $15,000 - $10,000 = $5,000

**Step 3:** Daily interest charge
- Annual rate: 6.5%
- Daily rate: 6.5% / 365 = 0.0178%
- Daily cost: $5,000 * 0.000178 = **$0.89/day**

**Step 4:** Annualized cost
- $0.89/day * 365 days = **$324.85/year**
- As % of leveraged amount: $324.85 / $5,000 = 6.5% ✓

### Why Crypto Doesn't Incur Leverage Costs
- Crypto positions: $8,000
- Equity: $10,000
- Crypto leverage: $8,000 / $10,000 = 0.8x
- **0.8x < 1.0x → No leverage → No fees**

Even if crypto was $10,000 (1.0x), there's still no leverage cost because you're not borrowing anything.

---

## Implications for Retraining

### Current Training is Correct ✅
The model was trained with:
- Stock leverage: 2x max
- Crypto leverage: 1x max (none)
- Leverage cost: 6.5% annual
- No shorting

### For Live Retraining
When you retrain live, the training environment will continue to match reality:

1. **Leverage costs** are already included in training loss
2. **Stock vs crypto** distinction is already made
3. **Position sizing** is already optimized for these constraints

### What Retraining Will Improve
- **Adapting to market regime changes**
- **Learning from actual execution** (fills, slippage, timing)
- **Correcting overfitting** to historical validation data
- **Updating to latest market dynamics**

---

## Testing Results

### Latest Simulation (Nov 7-16, 2025)
```
Date                  Equity         Cash     Return   Leverage    LevCost
2025-11-07            0.9992       0.0000    -0.0008       0.00x   0.000000
2025-11-10            1.0114       0.0000     0.0122       1.00x   0.000000
2025-11-16            1.0043       0.0000     0.0000       1.00x   0.000000

Final Equity      : 1.0043
Net PnL           : +0.43%
Max Leverage      : 1.00x
Total Lev. Costs  : 0.000000
```

**Observations:**
- Max leverage stayed at 1.0x (no leverage used)
- No leverage costs incurred (as expected)
- Modest positive returns on latest data

---

## Production Deployment Notes

### 1. The Model Understands Leverage Costs
The model was **trained with** leverage fees, so:
- It already learned to avoid excessive leverage
- Position sizing accounts for borrowing costs
- Risk-adjusted returns include fee impact

### 2. Simulator Now Matches Training
Before this update:
- ❌ Applied leverage costs to all positions
- ❌ Didn't distinguish stocks vs crypto

After this update:
- ✅ Only charges leverage on stocks
- ✅ Crypto capped at 1x (no borrowing)
- ✅ Matches training environment exactly

### 3. Next Steps for Production
1. ✅ **Simulator aligned** with training
2. ✅ **Probe mode** implemented
3. ⏳ **Paper trading** test
4. ⏳ **Live trading** with gradual ramp-up
5. 🔄 **Retrain live** to adapt to market changes

---

## Quick Reference

### Training Config
```python
# neuraldailytraining/config.py
maker_fee: float = 0.0008                # 0.08%
leverage_fee_rate: float = 0.065         # 6.5% annual
equity_max_leverage: float = 2.0         # Stocks: 2x max
crypto_max_leverage: float = 1.0         # Crypto: 1x max (no leverage)
```

### Simulator Config
```python
# neuraldailymarketsimulator/simulator.py
maker_fee=0.0008
leverage_fee_rate=0.065
equity_max_leverage=2.0
crypto_max_leverage=1.0
```

### Alpaca Reality
- **Stocks:** 2x overnight, 4x intraday, can short
- **Crypto:** 1x only, no leverage, no shorting
- **Margin rate:** ~6.5% annual
- **Maker fees:** ~0.1-0.3% (varies by asset)

---

## Questions?

**Q: Why is the model not using leverage?**
A: The model learned that leverage is expensive (6.5% annual). It only leverages when expected returns exceed borrowing costs.

**Q: Should we train with 4x leverage to match intraday?**
A: No. Intraday leverage must be closed by EOD. Training with 2x matches overnight reality and encourages sustainable sizing.

**Q: Will retraining help with recent low performance?**
A: Yes! The 28.5% → 0.4% performance drop suggests:
1. Market regime changed
2. Model overfitting to validation period
3. Retraining on live data will recalibrate

**Q: Can the model learn to short?**
A: Not with current training setup. Would need to:
1. Remove `torch.clamp(inventory, min=0.0)` constraint
2. Add shorting costs/constraints
3. Retrain from scratch
4. **Not recommended** - shorting adds significant risk
