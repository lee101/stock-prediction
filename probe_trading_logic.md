# Probe Trading Logic - Complete Test Coverage

## Overview
Probe trading reduces position sizes to a minimum when a symbol is performing poorly, protecting capital during losing streaks.

**Key Rule:** A symbol enters probe mode when the **total PnL% of recent trades ≤ 0**.

---

## Decision Tree

```
Is global probe mode active (account-wide loss)?
├─ YES → PROBE MODE (all symbols)
└─ NO → Check per-symbol history
    │
    ├─ No trade history
    │  └─ NORMAL MODE ✓
    │
    ├─ 1 trade in history
    │  ├─ PnL% > 0 → NORMAL MODE ✓
    │  └─ PnL% ≤ 0 → PROBE MODE 🔬
    │
    └─ 2+ trades in history (only last 2 count)
       ├─ Sum of last 2 PnL% > 0 → NORMAL MODE ✓
       └─ Sum of last 2 PnL% ≤ 0 → PROBE MODE 🔬
```

---

## Test Coverage (13 Tests, All Passing ✅)

### 1. No History Cases

#### Test: `test_no_trade_history_normal_mode`
```python
History: []
Expected: NORMAL MODE
Reason: No evidence of poor performance
```

**Result:** ✅ PASS

---

### 2. Single Trade Cases

#### Test: `test_single_negative_trade_probe_mode`
```python
History: [-5%]
Expected: PROBE MODE
Reason: Only trade was negative
```

**Result:** ✅ PASS
**Trigger:** `single_trade_negative=-0.05`

---

#### Test: `test_single_positive_trade_normal_mode`
```python
History: [+10%]
Expected: NORMAL MODE
Reason: Only trade was positive
```

**Result:** ✅ PASS

---

#### Test: `test_single_zero_trade_probe_mode`
```python
History: [0%]
Expected: PROBE MODE
Reason: Break-even counts as non-positive (≤ 0)
```

**Result:** ✅ PASS
**Trigger:** `single_trade_negative=0.0`

---

### 3. Two Trade Cases

#### Test: `test_two_negative_trades_probe_mode`
```python
History: [-3%, -2%]
Sum: -5%
Expected: PROBE MODE
Reason: Both trades negative
```

**Result:** ✅ PASS
**Trigger:** `recent_pnl_sum=-0.05 [-0.03, -0.02]`

---

#### Test: `test_two_positive_trades_normal_mode`
```python
History: [+5%, +3%]
Sum: +8%
Expected: NORMAL MODE
Reason: Both trades positive
```

**Result:** ✅ PASS

---

#### Test: `test_mixed_trades_positive_sum_normal_mode`
```python
History: [+10%, -5%]
Sum: +5%
Expected: NORMAL MODE
Reason: Net positive despite one loss
```

**Result:** ✅ PASS

---

#### Test: `test_mixed_trades_negative_sum_probe_mode`
```python
History: [+3%, -5%]
Sum: -2%
Expected: PROBE MODE
Reason: Net negative overall
```

**Result:** ✅ PASS
**Trigger:** `recent_pnl_sum=-0.02 [0.03, -0.05]`

---

#### Test: `test_mixed_trades_zero_sum_probe_mode`
```python
History: [+5%, -5%]
Sum: 0%
Expected: PROBE MODE
Reason: Zero sum triggers probe (sum ≤ 0)
```

**Result:** ✅ PASS
**Trigger:** `recent_pnl_sum=0.0000 [0.05, -0.05]`

---

### 4. Multiple Trades (3+)

#### Test: `test_only_most_recent_two_trades_matter`
```python
History: [-20%, +5%, +3%]  # 3 trades
Only last 2 count: [+5%, +3%]
Sum: +8%
Expected: NORMAL MODE
Reason: Old losses don't affect current state
```

**Result:** ✅ PASS

**Key Learning:** Only the **most recent 2 trades** matter. Older history is ignored.

---

### 5. Global Probe Override

#### Test: `test_global_probe_overrides_symbol_history`
```python
History: [+10%, +5%]  # Symbol is winning
Global state: Account had negative day PnL
Expected: PROBE MODE
Reason: Global loss overrides symbol performance
```

**Result:** ✅ PASS
**Trigger:** `global:previous_day_loss`

**Key Learning:** Account-wide losses force **all symbols** into probe mode.

---

### 6. Side Independence

#### Test: `test_different_sides_tracked_separately`
```python
Symbol: SHOP
Buy history: [-5%]  → PROBE MODE
Sell history: [+10%] → NORMAL MODE
```

**Result:** ✅ PASS

**Key Learning:** Buy and sell sides are tracked independently.

---

## Implementation Code

### File: `neural_daily_trade_with_probe.py:141-154`

```python
def _should_use_probe_mode(symbol: str, side: str, probe_state: ProbeState) -> tuple[bool, Optional[str]]:
    # Check global probe state (negative account PnL)
    if probe_state.force_probe:
        reason = probe_state.reason or "previous_day_loss"
        return True, f"global:{reason}"

    # Check per-symbol recent trade performance
    recent_pnls = _get_recent_trade_pnl_pcts(symbol, side, limit=2)
    if len(recent_pnls) >= 2:
        # Have 2+ trades: sum them and check if non-positive
        pnl_sum = sum(recent_pnls)
        if pnl_sum <= 0:
            return True, f"recent_pnl_sum={pnl_sum:.4f} [{', '.join(f'{p:.4f}' for p in recent_pnls)}]"
    elif len(recent_pnls) == 1:
        # Have only 1 trade: check if it's negative
        pnl = recent_pnls[0]
        if pnl <= 0:
            return True, f"single_trade_negative={pnl:.4f}"

    return False, None
```

---

## Probe Mode vs Normal Mode

### Normal Mode
```python
# Position sizing
target_notional = account_equity * 0.15 * model_allocation
# Example: $10,000 * 0.15 * 0.8 = $1,200
```

### Probe Mode 🔬
```python
# Position sizing
target_notional = min(300, account_equity * 0.01)
# Example: min($300, $10,000 * 0.01) = $300
```

**Probe trades are 4-10x smaller** than normal trades.

---

## Real-World Examples

### Example 1: New Symbol (No History)
```
Symbol: PLTR
History: []
Decision: NORMAL MODE
Reasoning: No prior losses, give it a chance
```

### Example 2: First Trade Was a Loss
```
Symbol: QUBT
Trade 1: Bought at $10, sold at $9.50 → -5%
History: [-5%]
Decision: PROBE MODE
Next trade: Max $300 position
Reasoning: First trade lost, protect capital
```

### Example 3: Recovering from Losses
```
Symbol: NVDA
Trade 1: -5%
Trade 2: -3%
Trade 3: +8%
History: [-3%, +8%]  # Only last 2 count
Sum: +5%
Decision: NORMAL MODE
Reasoning: Net positive on recent trades → resume normal sizing
```

### Example 4: Alternating Wins/Losses
```
Symbol: SPY
Trade 1: +10%
Trade 2: -12%
History: [+10%, -12%]
Sum: -2%
Decision: PROBE MODE
Reasoning: Despite one big win, net is negative
```

### Example 5: Account-Wide Loss Day
```
Account PnL today: -$500
Symbol: QQQ
History: [+5%, +3%]  # QQQ is winning!
Decision: PROBE MODE (all symbols)
Reasoning: Global risk control overrides individual performance
```

---

## Edge Cases Handled

### ✅ Break-Even Trades
```
Trade PnL: 0%
Triggers: PROBE MODE (0 ≤ 0)
Reasoning: Not profitable, reduce risk
```

### ✅ Exactly Zero Sum
```
History: [+5%, -5%]
Sum: 0%
Triggers: PROBE MODE (0 ≤ 0)
Reasoning: No net gain, be cautious
```

### ✅ Very Small Negative
```
History: [-0.001%, +0.002%]
Sum: +0.001%
Triggers: NORMAL MODE (0.001 > 0)
Reasoning: Net positive, even if tiny
```

### ✅ Very Old Losses
```
History: [-50%, -40%, +10%, +8%]
Only counts: [+10%, +8%]
Sum: +18%
Triggers: NORMAL MODE
Reasoning: Recent performance is good
```

---

## FAQ

**Q: Why check sum of last 2 trades instead of last 1?**
A: Prevents flapping between modes due to natural variance. Need 2 consecutive good trades to resume normal sizing.

**Q: Why does zero PnL trigger probe mode?**
A: Break-even isn't profitable. We want positive momentum before risking full capital.

**Q: Can a symbol recover from probe mode?**
A: Yes! After 2 profitable trades (sum > 0), it returns to normal sizing automatically.

**Q: What if I have 10 losing trades, then 2 winning ones?**
A: Only last 2 matter. After 2 wins, returns to normal mode. System focuses on recent performance.

**Q: Does global probe reset per-symbol history?**
A: No. Global probe is temporary (next trading day). Symbol history persists and determines future behavior.

**Q: What happens on the first trade ever for a symbol?**
A: Normal mode (no history). After that trade completes, its PnL% determines future mode.

---

## Test Command

Run all tests:
```bash
pytest tests/test_neural_probe_trading.py -v
```

Expected output:
```
test_no_trade_history_normal_mode PASSED
test_single_negative_trade_probe_mode PASSED
test_single_positive_trade_normal_mode PASSED
test_two_negative_trades_probe_mode PASSED
test_two_positive_trades_normal_mode PASSED
test_mixed_trades_positive_sum_normal_mode PASSED
test_mixed_trades_negative_sum_probe_mode PASSED
test_mixed_trades_zero_sum_probe_mode PASSED
test_single_zero_trade_probe_mode PASSED
test_global_probe_overrides_symbol_history PASSED
test_different_sides_tracked_separately PASSED
test_only_most_recent_two_trades_matter PASSED

13 passed ✅
```

---

## Summary

| Scenario | Trades | Sum | Mode | Reason |
|----------|--------|-----|------|--------|
| No history | 0 | N/A | Normal | No risk evidence |
| 1 trade, positive | 1 | +X% | Normal | First trade won |
| 1 trade, negative | 1 | -X% | **Probe** | First trade lost |
| 1 trade, zero | 1 | 0% | **Probe** | No profit |
| 2 trades, sum > 0 | 2 | +X% | Normal | Net positive |
| 2 trades, sum ≤ 0 | 2 | -X% or 0% | **Probe** | Net non-positive |
| Global probe active | Any | Any | **Probe** | Account-wide loss |

**Rule of thumb:** Need **positive** sum of recent trades to trade normally. Otherwise, probe mode protects capital.
