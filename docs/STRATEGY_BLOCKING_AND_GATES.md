# Strategy-Specific Blocking & Trade Gates

## Summary of Changes (2025-10-30)

### ✅ 1. Strategy-Specific Blocking
**Previously:** Blocks were per `symbol|side` (e.g., `ETHUSD|buy`)
**Now:** Blocks are per `symbol|side|strategy` (e.g., `ETHUSD|buy|maxdiff`)

**Benefits:**
- If maxdiff loses on ETHUSD-buy, only `ETHUSD-buy-maxdiff` gets blocked
- Other strategies (highlow, takeprofit, etc.) can still trade ETHUSD-buy independently
- Each strategy learns from its own performance

**Files Modified:**
- `src/trade_stock_state_utils.py`: Added `strategy` parameter to all state functions
- `trade_stock_e2e.py`: Updated block evaluation and outcome recording to use strategy

### ✅ 2. Disabled Volume Check
**Removed:** Minimum $5M daily dollar volume requirement (line 336-340)
**Reason:** Volume doesn't matter for your trading; strategy-specific blocks provide sufficient protection

### ✅ 3. Fixed Probe Trade Recording
**Fixed:** Trade outcomes now save with strategy-specific keys
**Fixed:** Learning state updates pass strategy parameter

## Default Behavior (No History)

✅ **Confirmed:** With no trade history, system defaults to **"normal"** trade mode (not probe)

```python
# trade_stock_e2e.py:1094
trade_mode = "probe" if (pending_probe or probe_active) else "normal"
```

When learning_state is empty:
- `pending_probe` = False
- `probe_active` = False
- Result: **normal trading** ✓

## Probe Trade Recording

✅ **Probe trades ARE properly recorded based on P&L:**

```python
# trade_stock_e2e.py:1031-1045
if trade_mode == "probe":
    _mark_probe_completed(symbol, side, successful=pnl_value > 0)  # ✓ Based on P&L
elif pnl_value < 0:
    _mark_probe_pending(symbol, side)  # Negative trade → enter probe mode
else:
    # Positive trade → clear probe flags
    _update_learning_state(..., pending_probe=False, probe_active=False)
```

**Note:** Probe functions currently don't support per-strategy probes yet (TODOs added at lines 1032, 1035)

## Trade Gates That Can Block

When `DISABLE_TRADE_GATES=0` (default), these gates can block trades:

### 1. **Spread Check** (line 1872-1873)
```python
if not tradeable:
    gating_reasons.append(spread_reason)
```
- Blocks if bid/ask spread is too wide or missing
- **Bypass:** Set `MARKETSIM_DISABLE_GATES=1`

### 2. **Edge Threshold** (line 1874-1875)
```python
if not edge_ok:
    gating_reasons.append(edge_reason)
```
- Blocks if expected move % is too small
- Controlled by symbol-specific thresholds

### 3. **Model Consensus** (line 1882-1883)
```python
if consensus_reason:
    gating_reasons.append(consensus_reason)
```
- Blocks if Toto and Kronos models disagree
- **Bypass:** Set to Kronos-only mode

### 4. **Loss Cooldown** (line 1884-1885)
```python
if not cooldown_ok and not kronos_only_mode:
    gating_reasons.append("Cooldown active after recent loss")
```
- Blocks for 3 days after a loss (per strategy now!)
- **Disabled in:** Kronos-only mode or `SIMPLIFIED_MODE=1`

### 5. **Kelly Fraction** (line 1886-1887)
```python
if kelly_fraction <= 0:
    gating_reasons.append("Kelly fraction <= 0")
```
- Blocks if Kelly sizing suggests zero position
- Based on edge strength and volatility

### 6. **Recent Strategy Returns** (line 1888-1892)
```python
recent_sum = strategy_recent_sums.get(best_strategy)
if recent_sum is not None and recent_sum <= 0:
    gating_reasons.append(f"Recent {best_strategy} returns sum {recent_sum:.4f} <= 0")
```
- Blocks if strategy has negative recent returns

### 7. **Strategy-Specific Loss Block** (line 1894-1900)
```python
base_blocked = block_info.get("blocked", False)
if base_blocked and block_info.get("block_reason"):
    combined_reasons.append(block_info["block_reason"])
```
- **Now strategy-specific!**
- Blocks only the specific symbol+side+strategy combination

## Environment Variables to Bypass Gates

```bash
# Disable all trade gates
MARKETSIM_DISABLE_GATES=1 python trade_stock_e2e.py

# Enable simplified mode (disables cooldown, probe trades)
SIMPLIFIED_MODE=1 python trade_stock_e2e.py

# Kronos-only mode (relaxes consensus and cooldown)
KRONOS_ONLY=1 python trade_stock_e2e.py
```

## Testing Recommendations

### Paper Trading Tests (PAPER=1)
```bash
# Test strategy-specific blocking
PAPER=1 python trade_stock_e2e.py

# Test with gates disabled
PAPER=1 MARKETSIM_DISABLE_GATES=1 python trade_stock_e2e.py

# Test simplified mode
PAPER=1 SIMPLIFIED_MODE=1 python trade_stock_e2e.py
```

### Unit Tests
```python
# Test that maxdiff block doesn't affect highlow
from src.trade_stock_state_utils import state_key

# Different strategies get different keys
key1 = state_key("ETHUSD", "buy", "maxdiff")   # ETHUSD|buy|maxdiff
key2 = state_key("ETHUSD", "buy", "highlow")   # ETHUSD|buy|highlow
assert key1 != key2  # ✓ Independent blocking
```

## Alternate Implementation: predict_stock_e2e.py

`predict_stock_e2e.py` appears to be a simpler/older implementation without:
- Strategy-specific blocking
- Probe trades
- Complex gating logic

Consider it as a reference or fallback for simpler trading logic.

## What Could Block Money-Making?

**Potential Issues:**
1. ✅ ~~Volume check~~ - **FIXED** (disabled)
2. ✅ ~~Global symbol+side blocking~~ - **FIXED** (now per-strategy)
3. ⚠️ **Probe mode not strategy-aware** - TODOs added, but probes still block at symbol+side level
4. ⚠️ **Consensus check** - May block if Toto and Kronos disagree (bypass with Kronos-only mode)
5. ⚠️ **Kelly fraction = 0** - Will block if edge is too small
6. ⚠️ **Recent negative returns** - Will block if strategy has recent losses

**Recommended Settings for Max Trading Freedom:**
```bash
PAPER=1 \
MARKETSIM_DISABLE_GATES=1 \
SIMPLIFIED_MODE=1 \
python trade_stock_e2e.py
```

This disables most gates while keeping strategy-specific learning intact.
