# Close Policy Analysis: INSTANT_CLOSE vs KEEP_OPEN

## Executive Summary

After running comprehensive simulations comparing instant-close (close unfilled positions at EOD) vs keep-open (let positions persist multi-day) strategies, we have clear recommendations:

### Recommendations by Asset Class

| Asset Type | Policy | Rationale |
|------------|--------|-----------|
| **Crypto (BTC/ETH)** | INSTANT_CLOSE | Taker fees (0.25%) < opportunity cost. Negative returns when keeping open. |
| **Stocks (all)** | KEEP_OPEN | Zero taker fees. Shorts benefit 3-5x more than longs from patience. |

---

## Methodology

### Test Setup
- Simulations per symbol: 10-50
- Strategy tested: maxdiffalwayson (market-neutral, both long and short)
- Data source: Historical backtests from 2024-09-07
- Fee modeling:
  - Crypto: 0.25% taker fee (Alpaca)
  - Stocks: 0% taker fee + minimal SEC fees
  - Opportunity cost: 5% annual return on tied-up capital

### Key Metrics
- **Gross Return**: Total profit before fees
- **Close Fees**: Fees paid for closing unfilled positions at market (instant_close only)
- **Opportunity Cost**: Cost of capital tied up in unfilled positions (keep_open only)
- **Net Return**: Gross return - close fees - opportunity cost

---

## Results

### Crypto: INSTANT_CLOSE Wins

#### BTCUSD (10 simulations)
```
Policy          Net Return    Close Fees   Opp Cost   Advantage
instant_close    +5.78%       0.0100       0.0000
keep_open       -10.33%       0.0000       0.0005
Winner: INSTANT_CLOSE (+16.11%)
```

#### ETHUSD (10 simulations)
```
Policy          Net Return    Close Fees   Opp Cost   Advantage
instant_close    +5.59%       0.0095       0.0000
keep_open        -6.13%       0.0000       0.0005
Winner: INSTANT_CLOSE (+11.72%)
```

**Insight**: For crypto, the negative returns from keeping positions open far outweigh the 0.25% taker fees. The maxdiffalwayson strategy's predictions don't fill reliably enough to justify holding crypto positions multi-day.

---

### Stocks: KEEP_OPEN Wins (Dominated by Short Side)

#### GOOG (5-10 simulations)
```
Overall:
Policy          Net Return    Close Fees   Opp Cost   Advantage
instant_close    -3.24%       0.0000       0.0000
keep_open        +0.90%       0.0000       0.0010
Winner: KEEP_OPEN (+4.14%)

Side Breakdown:
                Buy Return    Sell Return
instant_close    +0.63%       -3.87%
keep_open        -1.22%       +2.13%

Side Advantage:
Buy (long):   -1.86%  (worse with KEEP_OPEN)
Sell (short): +6.00%  (better with KEEP_OPEN)
→ Sell side benefits 3.2x more from KEEP_OPEN
```

#### META (5-10 simulations)
```
Overall:
Policy          Net Return    Close Fees   Opp Cost   Advantage
instant_close    +4.50%       0.0000       0.0000
keep_open        +9.10%       0.0000       0.0007
Winner: KEEP_OPEN (+4.60%)

Side Breakdown:
                Buy Return    Sell Return
instant_close    +6.19%       -1.69%
keep_open        +5.12%       +3.98%

Side Advantage:
Buy (long):   -1.07%  (worse with KEEP_OPEN)
Sell (short): +5.68%  (better with KEEP_OPEN)
→ Sell side benefits 5.3x more from KEEP_OPEN
```

**Critical Insight**: KEEP_OPEN for stocks is primarily beneficial for **short positions**. Longs actually do slightly worse with KEEP_OPEN, but since maxdiffalwayson does both sides, and the short side benefit (3-5x) dominates, KEEP_OPEN is the right overall policy.

---

## Why Shorts Benefit More

### Short Position Dynamics
1. **Entry**: Sell at `high_pred` (predicted high price)
2. **Exit target**: Buy back at `low_actual` (actual low price)
3. **Problem**: If price doesn't drop to `low_actual` by EOD:
   - **INSTANT_CLOSE**: Close at `close_actual` → Often locks in a **loss** (bought back higher than expected)
   - **KEEP_OPEN**: Wait for price to drop → Higher chance of hitting profitable exit

### Long Position Dynamics
1. **Entry**: Buy at `low_pred`
2. **Exit target**: Sell at `high_actual`
3. **Less asymmetry**: Stock drift is generally upward, so instant close doesn't hurt as much

---

## Implementation

### Storage Layer
- Location: `hyperparams/close_policy/{SYMBOL}.json`
- Functions: `save_close_policy()`, `load_close_policy()`
- Format:
```json
{
  "symbol": "GOOG",
  "close_policy": "KEEP_OPEN",
  "comparison": {
    "instant_close_net_return": -3.2398,
    "keep_open_net_return": 0.9029,
    "advantage": 4.1427,
    "buy_advantage": -1.8572,
    "sell_advantage": 6.0009
  }
}
```

### Resolution Function
```python
from backtest_test3_inline import resolve_close_policy

policy = resolve_close_policy("BTCUSD")  # Returns: "INSTANT_CLOSE"
policy = resolve_close_policy("GOOG")    # Returns: "KEEP_OPEN"
```

**Defaults** (if no stored policy):
- Crypto → INSTANT_CLOSE
- Stocks → KEEP_OPEN

### Automatic Evaluation
When running `backtest_test3_inline.py`, the system automatically:
1. Runs 10 comparison simulations
2. Determines the best policy
3. Saves to `hyperparams/close_policy/`

---

## Future Enhancements

### Potential Side-Specific Policies
Given the strong asymmetry between long and short performance, future work could explore:
- **INSTANT_CLOSE for longs** (buy side)
- **KEEP_OPEN for shorts** (sell side)

This would require:
1. Watcher spawn differentiation by side
2. Separate policy storage per side
3. Testing to confirm the combined benefit exceeds current KEEP_OPEN-for-both

### Per-Symbol Calibration
Current system automatically calibrates per symbol during backtesting. Symbols with:
- High volatility → May benefit more from instant close
- Strong directional bias → May show different long/short asymmetry
- Low liquidity → Limit orders less likely to fill, favoring instant close

---

## Related Files

### Core Implementation
- `hyperparamstore/store.py` - Storage functions
- `backtest_test3_inline.py:1655` - `resolve_close_policy()`
- `backtest_test3_inline.py:2974` - `evaluate_and_save_close_policy()`
- `test_backtest4_instantclose_inline.py` - Comparison logic

### Analysis
- `docs/CRYPTO_BACKOUT_FEE_ANALYSIS.md` - Original crypto fee analysis
- `test_backtest4_output_v2.log` - Full simulation results

---

## Conclusions

1. ✅ **Crypto**: INSTANT_CLOSE is optimal (avoid long holds with negative returns)
2. ✅ **Stocks**: KEEP_OPEN is optimal (primarily due to short side benefit)
3. 🔍 **Asymmetry discovered**: Shorts benefit 3-5x more from KEEP_OPEN than longs
4. 🚀 **Future opportunity**: Side-specific policies could further optimize returns

The system is now production-ready with automatic policy evaluation and storage.
