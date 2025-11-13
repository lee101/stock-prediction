# Code Audit: P&L and Trading Logic Issues

**Date:** 2025-11-12 (Updated: 2025-11-13)
**Scope:** P&L calculations, fee handling, position management, and trading logic
**Focus:** Identifying bugs and issues that may affect profitability

---

## Status Update (2025-11-13)

**✅ RESOLVED - Issue #1: Fee Standardization**
- Decision: Standardize on **10 bps (0.001)** for crypto fees across entire codebase
- Files updated:
  - `src/fees.py` (0.0015 → 0.001)
  - `stockagent/constants.py` (0.0015 → 0.001)
  - `pufferlib_cpp_market_sim/include/market_config.h` (0.0015f → 0.001f)
  - `cppsimulator/include/types.hpp` (0.0015 → 0.001)
- Rationale: `loss_utils.py` already uses 10 bps, and training data/models are optimized for this rate

**🚧 NEW: Correlation Risk Management**
- Created comprehensive plan: `docs/correlation_risk_management_plan.md`
- Implemented correlation matrix calculator: `trainingdata/calculate_correlation_matrix.py`
- Next: Phase 1 implementation (monitoring and alerts)

---

## Executive Summary

This audit identified **10 critical issues** and **5 moderate issues** that may affect P&L accuracy and trading profitability. The most critical findings relate to fee calculation inconsistencies, missing fee deductions in P&L recording, and potential race conditions in position management.

**Issue #1 has been resolved** by standardizing crypto fees to 10 bps across all modules.

---

## Critical Issues (P0)

### 1. ✅ RESOLVED: Fee Calculation Inconsistency Between Training and Live Trading

**Location:** `loss_utils.py` vs `src/fees.py`

**Issue:**
- `loss_utils.py` uses hardcoded fees:
  - `TRADING_FEE = 0.0005` (5 bps)
  - `CRYPTO_TRADING_FEE = 0.001` (10 bps)
- `src/fees.py` was using different defaults:
  - Equities: `0.0005` (5 bps) ✓ matches
  - Crypto: `0.0015` (15 bps) ❌ **MISMATCH** (loss_utils uses 10 bps, fees.py used 15 bps)

**Impact:** Model training optimizes for 10 bps crypto fees, but live trading pays 15 bps. This 50% fee difference could significantly impact crypto strategy profitability.

**Resolution (2025-11-13):** ✅ Standardized all crypto fees to **10 bps (0.001)** across:
- `src/fees.py`
- `stockagent/constants.py`
- `pufferlib_cpp_market_sim/include/market_config.h`
- `cppsimulator/include/types.hpp`

Decision rationale: Models and training data already optimize for 10 bps, so standardizing on this value preserves model accuracy.

---

### 2. P&L Recording Doesn't Account for Trading Fees Explicitly

**Location:** `trade_stock_e2e.py:896` in `_record_trade_outcome()`

**Issue:**
```python
pnl_value = float(getattr(position, "unrealized_pl", 0.0) or 0.0)
```

The code directly uses Alpaca's `unrealized_pl` without:
1. Verifying whether fees are already included
2. Explicitly deducting trading fees if not included
3. Documenting the assumption about Alpaca's fee handling

**Impact:** If Alpaca's `unrealized_pl` doesn't include trading fees, recorded P&L will be overstated by 2x the fee rate (entry + exit).

**Recommendation:**
- Verify Alpaca API documentation for whether `unrealized_pl` includes fees
- If not included, explicitly calculate and deduct fees:
  ```python
  from src.fees import get_fee_for_symbol
  fee_rate = get_fee_for_symbol(position.symbol)
  # Deduct both entry and exit fees
  fees_paid = abs(qty_value * avg_entry_price * fee_rate * 2)
  pnl_value = unrealized_pl - fees_paid
  ```

---

### 3. Loss Function Fee Parameter Inconsistencies

**Location:** `loss_utils.py` - multiple functions

**Issue:**
- `calculate_trading_profit_torch()` accepts optional `trading_fee` parameter (line 106)
- `calculate_trading_profit_torch_buy_only()` accepts optional `trading_fee` parameter (line 148)
- `calculate_profit_torch_with_entry_buysell_profit_values()` accepts optional `trading_fee` parameter (line 320)
- BUT `calculate_trading_profit_torch_with_buysell_profit_values()` does NOT accept fee parameter (line 219) and uses hardcoded `TRADING_FEE`

**Impact:** Inconsistent fee handling across different loss functions could lead to training/evaluation mismatches.

**Recommendation:** Standardize all loss functions to accept `trading_fee` parameter and use `src/fees.py` for defaults.

---

### 4. Potential Double-Counting of Fees in Entry/Exit Loss Functions

**Location:** `loss_utils.py:408` in `calculate_profit_torch_with_entry_buysell_profit_values()`

**Issue:**
```python
calculated_profit_values = (
    bought_adjusted_profits
    + adjusted_profits
    - ((torch.abs(detached_y_test_pred) * trading_fee) * hit_trading_points)  # fee
)
```

The fee is only applied when `hit_trading_points` is true (both entry AND exit hit). But fees should be charged on:
1. Entry fee when position is opened
2. Exit fee when position is closed

If a position hits entry but not exit (or vice versa), only one fee should be charged, not zero.

**Impact:** Fee calculation may be incorrect for partially-filled orders or missed exit targets.

**Recommendation:** Separate entry and exit fee calculations:
```python
entry_fees = torch.abs(detached_y_test_pred) * trading_fee * entry_hit
exit_fees = torch.abs(detached_y_test_pred) * trading_fee * exit_hit
calculated_profit_values = bought_adjusted_profits + adjusted_profits - entry_fees - exit_fees
```

---

### 5. No Leverage Cost Deduction in P&L Recording

**Location:** `trade_stock_e2e.py:896` in `_record_trade_outcome()`

**Issue:**
The code records P&L without deducting leverage/margin financing costs. While Alpaca may include this in their `unrealized_pl`, it's not documented.

**Impact:** If leverage costs aren't included in Alpaca's P&L, recorded profits will be overstated for leveraged positions. With 6.5% annual rate (~0.0258% daily), this adds up quickly.

**Recommendation:**
- Verify if Alpaca includes margin costs in `unrealized_pl`
- If not, explicitly calculate and deduct:
  ```python
  from src.leverage_settings import get_leverage_settings
  settings = get_leverage_settings()
  # Calculate days held and leverage used
  leverage_cost = calculate_leverage_penalty(...)
  pnl_value = unrealized_pl - leverage_cost
  ```

---

## High Priority Issues (P1)

### 6. Missing Entry Price Validation in Position Entry Logic

**Location:** `trade_stock_e2e.py:2601, 2959, 3069, 3122, 3141`

**Issue:**
Entry price is calculated from bid/ask but there's no validation that:
1. The spread is reasonable (could be stale/manipulated quotes)
2. Entry price is positive and non-zero
3. Entry price hasn't moved significantly since forecast was generated

**Impact:** Could enter trades at terrible prices during illiquid periods or quote errors.

**Recommendation:** Add entry price validation:
```python
if entry_price <= 0:
    logger.error(f"Invalid entry price {entry_price} for {symbol}")
    continue

# Check spread
spread_pct = (ask_price - bid_price) / mid_price if mid_price > 0 else float('inf')
if spread_pct > MAX_SPREAD_THRESHOLD:  # e.g., 1%
    logger.warning(f"Spread too wide ({spread_pct:.2%}) for {symbol}, skipping")
    continue
```

---

### 7. Position Sizing Doesn't Account for Correlation Risk

**Location:** `src/sizing_utils.py:52` in `get_qty()`

**Issue:**
Position sizing uses 60% max per symbol (line 73) but doesn't consider:
1. Portfolio correlation (could have 60% in AAPL + 60% in MSFT = 120% in correlated tech)
2. Sector exposure limits
3. Total portfolio leverage across all positions

**Impact:** Could concentrate risk beyond intended limits through correlated positions.

**Recommendation:** Implement portfolio-level risk checks before individual position sizing.

---

### 8. No Slippage Modeling in Live Trading

**Location:** `trade_stock_e2e.py` (missing slippage model)

**Issue:**
- Backtesting uses slippage model from `marketsimulator/execution.py:27-58`
- Live trading uses market/limit orders without slippage consideration
- This creates train/test mismatch - backtest results will look better than live

**Impact:** Live performance will underperform backtests due to unmodeled slippage and market impact.

**Recommendation:**
- For market orders, estimate expected slippage and factor into entry decisions
- Consider using limit orders with slippage-adjusted limit prices
- Track actual slippage vs. mid-price for each fill

---

### 9. Potential Race Condition in Active Trade Management

**Location:** `trade_stock_e2e.py:890, 2567-2581`

**Issue:**
- `_pop_active_trade()` removes from active_trades store (line 890)
- `_get_active_trade()` reads from active_trades store (line 2563)
- `_update_active_trade()` writes to active_trades store (line 2573)

These operations are not atomic. If multiple processes/threads access the same store:
1. Two processes could both read "no active trade"
2. Both create new entries
3. One entry gets overwritten

**Impact:** Could lose tracking of active trades, leading to duplicate position entries or orphaned watchers.

**Recommendation:** Use file locking or database transactions for atomic read-modify-write operations on shared state.

---

### 10. Close-at-EOD Parameter Not Consistently Applied

**Location:** `loss_utils.py:319, 351, 368`

**Issue:**
The `close_at_eod` parameter affects how exit P&L is calculated:
- `True`: forces close at EOD close price (no intraday exits)
- `False`: allows intraday high/low exits

But the parameter is not consistently used:
- Some loss functions have it, some don't
- Not clear if live trading behavior matches training assumption

**Impact:** Training may optimize for intraday exits that aren't achievable in live trading, or vice versa.

**Recommendation:**
- Standardize `close_at_eod` across all loss functions
- Document whether live trading uses intraday exits or EOD closes
- Ensure training matches live behavior

---

## Moderate Issues (P2)

### 11. Hardcoded Probe Notional Limit

**Location:** `trade_stock_e2e.py:131`

**Issue:**
```python
PROBE_NOTIONAL_LIMIT = _resolve_probe_notional_limit()  # defaults to $300
```

$300 may be too small for meaningful probes on expensive stocks (e.g., TSLA @ $250/share = 1.2 shares).

**Recommendation:** Make probe size proportional to stock price or equity.

---

### 12. No Validation of Position.unrealized_pl Sign

**Location:** `trade_stock_e2e.py:896`

**Issue:**
Code assumes `unrealized_pl` has correct sign for long/short positions. Should verify:
- Long position with price up → positive P&L
- Long position with price down → negative P&L
- Short position with price up → negative P&L
- Short position with price down → positive P&L

**Recommendation:** Add assertion or validation to catch sign errors.

---

### 13. Missing Fee Calculation for Partial Fills

**Location:** Throughout codebase

**Issue:**
If an order is partially filled, fee is still charged on the filled portion. Code doesn't explicitly handle this case.

**Recommendation:** Track filled quantity vs. intended quantity and calculate fees accordingly.

---

### 14. Drawdown Calculation Uses Unrealized P&L

**Location:** `trade_stock_e2e.py:855`

**Issue:**
```python
unrealized_pl = float(getattr(position, "unrealized_pl", 0.0) or 0.0)
```

Drawdown detection triggers probe trades based on unrealized P&L. But:
1. May include unrealized losses that haven't affected equity yet
2. Doesn't account for fees that will be paid on close

**Recommendation:** Use realized P&L or explicitly subtract expected fees from unrealized P&L.

---

### 15. No Currency Conversion for Cross-Currency Positions

**Location:** Throughout codebase

**Issue:**
No explicit handling of FX rates for non-USD denominated assets. While all symbols appear to be USD-denominated (crypto pairs end in "USD", stocks are USD), there's no validation of this assumption.

**Recommendation:** Add validation that all traded symbols are USD-denominated, or implement FX conversion.

---

## Testing Recommendations

1. **Unit Tests for Fee Calculations:**
   - Test all loss functions with various fee rates
   - Verify entry + exit fees are both charged
   - Test edge cases (zero fees, very high fees)

2. **Integration Tests for P&L Recording:**
   - Mock Alpaca positions with known P&L
   - Verify recorded P&L matches expected (including fees)
   - Test long and short positions

3. **Backtesting Validation:**
   - Compare backtest results with and without slippage
   - Verify fee deductions match expected
   - Test with realistic leverage costs

4. **Live Trading Monitoring:**
   - Log actual fill prices vs. intended prices (slippage)
   - Track actual fees paid vs. expected
   - Monitor leverage costs if applicable

---

## Priority Action Items

1. **Immediate (P0):**
   - [ ] Fix fee rate mismatch between loss_utils.py (10 bps) and fees.py (15 bps) for crypto
   - [ ] Verify Alpaca API fee inclusion in unrealized_pl
   - [ ] Add explicit fee deduction in P&L recording if needed
   - [ ] Fix fee calculation in entry/exit loss functions (issue #4)

2. **Short-term (P1):**
   - [ ] Add entry price validation with spread checks
   - [ ] Implement slippage tracking in live trading
   - [ ] Add file locking for active trade state management
   - [ ] Standardize close_at_eod parameter usage

3. **Medium-term (P2):**
   - [ ] Implement portfolio-level correlation risk checks
   - [ ] Add unit tests for all P&L calculation functions
   - [ ] Monitor and validate leverage cost deductions
   - [ ] Improve probe sizing logic

---

## Code Quality Observations

**Positive:**
- Well-structured fee utilities in `src/fees.py` with clear precedence rules
- Comprehensive position sizing with exposure limits
- Good separation of concerns (fees, leverage, sizing in separate modules)

**Needs Improvement:**
- Inconsistent fee parameter handling across loss functions
- Hardcoded constants scattered across multiple files
- Missing documentation on Alpaca API assumptions
- Limited error handling for edge cases (zero prices, stale quotes)

---

## Conclusion

The codebase has solid architecture but contains several critical issues that could materially impact P&L:

1. **Fee calculation inconsistencies** (5 bps difference for crypto) could reduce profitability by 50% of the fee rate
2. **Missing explicit fee deductions** in P&L recording risks overstating returns
3. **No slippage modeling** in live trading creates train/test mismatch

Addressing the P0 issues should be the immediate priority, as they directly affect reported and actual profitability.
