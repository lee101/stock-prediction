# Work Stealing Simulator - Preliminary Findings

## Issue Discovered

All optimization runs show:
- `steals: 0`
- `blocks: 0`

This means work stealing logic is never triggered because capital capacity is never exceeded.

## Current Simulation Parameters

```python
max_leverage = 2.0
capital = $10,000
position_size = $1,000 per order
sampling = every 168 hours (weekly)
cryptos = 5 available per timestep
```

Maximum capital used: $10,000 * 2.0 = $20,000
Maximum concurrent positions: 20,000 / 1,000 = **20 positions**

But only attempting 5 crypto orders per week, so capacity is never hit.

## Best Performing Configs (Without Work Stealing)

Top scoring configs around 221k-222k:

1. **Config [1.28, 0.040, 0.015, 0.015, ...]**
   - Total PnL: $440,041
   - Sharpe: 0.526
   - Win rate: 54.7%
   - Trades: 1,106

2. **Config [1.92, 0.046, 0.015, 0.008, ...]**
   - Total PnL: $440,398
   - Sharpe: 0.526  
   - Win rate: 54.8%
   - Trades: 1,109

3. **Config [2.99, 0.046, 0.013, 0.006, ...]**
   - Total PnL: $439,891
   - Sharpe: 0.534
   - Win rate: 54.9%
   - Trades: 1,074

## Required Fixes

1. **Reduce capital** to $5,000 or lower
2. **Increase position size** to $2,000-$3,000
3. **Sample more frequently** (every 24-48 hours instead of weekly)
4. **Attempt more concurrent orders** (10+ cryptos instead of 5)

This will create capacity pressure and actually test work stealing parameters.

## Test Observations

Configs with lower scores (15k-22k range) have:
- Much lower PnL ($18k-$37k vs $440k)
- Higher Sharpe (2.2-2.8 vs 0.52-0.54)
- Similar win rates (~54%)
- Fewer trades (549-998 vs 1,074-1,109)

The scoring function weights PnL heavily, so high-frequency trading dominates.
