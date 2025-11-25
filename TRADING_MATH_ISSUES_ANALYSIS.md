# Trading Math and CI Issues Analysis

**Date:** 2025-11-24
**Branch:** claude/fix-trading-math-ci-01GZgfpAqWAYJtU88GPotzSv

## Executive Summary

Found **3 critical issues** with trading math and MarketSimulator alignment, plus **2 CI modernization opportunities**.

## Critical Issues Found

### 1. **CRITICAL: Incorrect Trading Fee for Equities** 🔴

**Location:** `loss_utils.py:8-9`

**Issue:**
```python
TRADING_FEE = 0.0005
# equities .0000278
```

The trading fee is set to **0.0005 (5 bps)** but the comment indicates it should be **0.0000278 (0.278 bps)**. This is an **18x difference**!

**Impact:**
- Training models assume 18x higher costs than production
- This causes significant misalignment between backtests and live trading
- Models trained with this fee will be overly conservative

**Evidence:**
- Alpaca offers commission-free trading (effectively 0 bps for equities)
- Modern brokers like IBKR charge ~0.25-0.5 bps
- The comment in line 9 shows the intended value

**Root Cause:**
- Likely a typo or legacy value that wasn't updated
- The commented values above (lines 3-4) show historical fee exploration

---

### 2. **CRITICAL: Missing Leverage Costs in Loss Functions** 🔴

**Location:** `loss_utils.py` - all `calculate_*_profit` functions

**Issue:**
The loss functions (`calculate_trading_profit_torch`, `calculate_profit_torch_with_entry_buysell_profit_values`, etc.) do NOT account for leverage financing costs, but the MarketSimulator DOES.

**MarketSimulator (CORRECT):**
```python
# marketsimulator/state.py:262-263
daily_rate = self.leverage_settings.daily_cost  # 0.065 / 252 = 0.000257936
cost = borrow_notional * daily_rate * day_fraction
```

**Loss Functions (MISSING):**
- No leverage cost calculation
- Only trading fees are deducted
- This means training doesn't penalize leverage use properly

**Impact:**
- Models trained with loss_utils will overuse leverage
- Production (MarketSimulator) will have worse performance due to financing costs
- Significant train/prod skew, especially for leveraged positions

**Example:**
- 2x leveraged position held for 1 day on $100k
- Borrow amount: $100k
- Daily cost: $100k × 0.000257936 = **$25.79/day**
- This cost is **missing** from training but **present** in production

---

### 3. **Medium: Inconsistent Fee Application Across Functions** 🟡

**Location:** `loss_utils.py` - multiple functions

**Issue:**
Several functions have hardcoded `TRADING_FEE` instead of accepting it as a parameter:

- `get_trading_profits_list()` - line 507
- `calculate_trading_profit_torch_with_buysell_profit_values()` - line 265

Meanwhile, newer functions properly accept `trading_fee` parameter:
- `calculate_trading_profit_torch()` - line 106
- `calculate_profit_torch_with_entry_buysell_profit_values()` - line 320

**Impact:**
- Cannot test different fee scenarios in some functions
- Crypto vs equity fee differentiation is inconsistent
- Test coverage gaps

---

## CI/Modernization Issues

### 4. **CI: Python 3.13 Not Fully Tested** 🟡

**Location:** `.github/workflows/ci.yml` and `.github/workflows/ci-fast.yml`

**Current State:**
- Full CI uses Python 3.13
- Fast CI uses Python 3.13
- Local development uses Python 3.11.14 (per current environment)

**Issue:**
- Matrix only tests one Python version (3.13)
- Should test 3.11+ for compatibility
- No Python 3.12 testing

**Recommendation:**
```yaml
strategy:
  matrix:
    python-version: ["3.11", "3.12", "3.13"]
```

---

### 5. **CI: Missing Dependency Caching** 🟡

**Location:** `.github/workflows/ci-fast.yml:79`

**Current:**
```yaml
- name: Set up Python ${{ matrix.python-version }}
  uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
    cache: 'pip'
```

**Issue:**
- Caches `pip` but uses `uv` for installation
- Should cache `uv` dependencies instead

**Recommendation:**
```yaml
cache: 'pip'
cache-dependency-path: |
  requirements.txt
  requirements-ci.txt
```

Or use uv's built-in caching.

---

## Recommended Fixes

### Priority 1: Fix Trading Fees

```python
# loss_utils.py
TRADING_FEE = 0.0000278  # Actual equity commission (0.278 bps)
CRYPTO_TRADING_FEE = 0.008  # 80 bps crypto fee (maker/taker normalized)
```

**Alternative (More Accurate):**
```python
# For Alpaca commission-free trading
TRADING_FEE = 0.00  # Alpaca: commission-free
# But account for SEC fees, exchange fees, etc.
EQUITY_REGULATORY_FEE = 0.0000278  # ~$0.278 per $1000
CRYPTO_TRADING_FEE = 0.008
```

### Priority 2: Add Leverage Costs to Loss Functions

**Option A: Add to existing functions (backward compatible)**
```python
def calculate_trading_profit_torch(
    scaler, last_values, y_test, y_test_pred,
    trading_fee=None,
    leverage_cost_daily=None,  # NEW
    position_duration_days=1.0,  # NEW
):
    if trading_fee is None:
        trading_fee = TRADING_FEE
    if leverage_cost_daily is None:
        leverage_cost_daily = 0.000257936  # 6.5% annual / 252 days

    # Calculate leverage cost for positions > 1.0
    leverage_penalty = torch.clamp(torch.abs(y_test_pred) - 1.0, min=0.0) * leverage_cost_daily * position_duration_days

    # ... existing logic ...

    current_profit = (...) - leverage_penalty
```

**Option B: Create new functions with leverage (recommended)**
```python
def calculate_trading_profit_torch_with_leverage(
    scaler, last_values, y_test, y_test_pred,
    trading_fee=None,
    leverage_settings=None,
    position_duration_days=1.0,
):
    """
    Calculate trading profit accounting for both fees and leverage costs.
    Aligns with MarketSimulator production behavior.
    """
    ...
```

### Priority 3: Standardize Fee Parameters

Update all functions to accept `trading_fee` parameter:
- `get_trading_profits_list()`
- `calculate_trading_profit_torch_with_buysell_profit_values()`

---

## Testing Requirements

1. **Update test_loss_utils.py:**
   - Add tests with correct fee values
   - Add tests for leverage cost calculations
   - Add tests comparing with MarketSimulator

2. **Create integration test:**
   ```python
   def test_loss_utils_matches_marketsimulator():
       """Verify loss_utils and MarketSimulator produce same PnL"""
       # Run same trades through both systems
       # Assert PnL matches within tolerance
   ```

3. **Regression tests:**
   - Ensure existing tests still pass with fee changes
   - May need to update expected values

---

## Migration Path

1. **Phase 1: Fix fees (backward compatible)**
   - Update TRADING_FEE constant
   - Add new constants for different fee tiers
   - Keep old functions working

2. **Phase 2: Add leverage costs (new functions)**
   - Create `*_with_leverage()` variants
   - Deprecate old functions
   - Update training code to use new functions

3. **Phase 3: Modernize CI**
   - Add Python version matrix
   - Optimize caching
   - Add integration tests

---

## Files Requiring Changes

### Immediate Fixes:
- `loss_utils.py` - fee constants and leverage cost logic
- `test_loss_utils.py` - update expected values
- `.github/workflows/ci.yml` - add test matrix
- `.github/workflows/ci-fast.yml` - fix caching

### Documentation:
- `README_CONFIGURATION.txt` - document fee settings
- `marketsimulator/QUICKSTART_SIZING.md` - clarify fee assumptions

---

## Risk Assessment

**High Risk:**
- Changing TRADING_FEE will affect all trained models
- Need to retrain or adjust model predictions

**Medium Risk:**
- Adding leverage costs changes optimization landscape
- Existing strategies may perform differently

**Low Risk:**
- CI changes are isolated
- New functions don't break old code

---

## Next Steps

1. ✅ Complete analysis (this document)
2. ⏭️ Fix TRADING_FEE constant
3. ⏭️ Add leverage cost functions
4. ⏭️ Update tests
5. ⏭️ Modernize CI
6. ⏭️ Run full test suite
7. ⏭️ Commit and push
8. ⏭️ Create PR with detailed changelog

---

## References

- `loss_utils.py` - Trading profit calculations
- `marketsimulator/state.py` - Production simulator with correct fees/costs
- `src/leverage_settings.py` - Centralized leverage configuration
- `marketsimulator/sizing_strategies.py` - Position sizing with leverage costs
- Alpaca API docs - Commission-free trading model
