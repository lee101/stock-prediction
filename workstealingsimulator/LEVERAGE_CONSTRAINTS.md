# Realistic Leverage Constraints

## Capital and Leverage Rules

### Stock Trading
- **Max intraday leverage**: 4x
- **Max end-of-day leverage**: 2x (must close positions to meet this)
- **Leverage fee**: 6.5% annual on borrowed portion (amount > 1x capital)
  - Applied at market open
  - Only on leveraged portion (capital_used - base_capital)
- **Work stealing applies**: Yes, stocks compete for 4x intraday capacity

### Crypto Trading
- **Leverage**: None (1x only)
- **No leverage fee**: Crypto positions are cash-backed
- **Work stealing applies**: Yes, but within 1x capital limit
- **Available pairs**: Limited (~3-5 pairs)

## Portfolio Composition

With $10k capital:

### Crypto Only
- Max exposure: $10k (1x, no leverage)
- 2 crypto orders @ $5k each = 100% of portfolio
- No leverage fees
- Capacity constraint is much tighter

### Stocks Only
- Max intraday exposure: $40k (4x)
- Max EOD exposure: $20k (2x)
- Leverage fee on $30k borrowed (at 4x) = $2,025/day
- Leverage fee on $10k borrowed (at 2x) = $675/day
- 2 stock orders @ $20k each = 4x leverage (hit max)

### Mixed Portfolio
- Crypto uses up to 1x capital (no leverage)
- Stocks can leverage remaining + up to 4x total
- Example: $5k in crypto, $35k in stocks = 4x total

## Simulation Implications

### Current Issue
Simulator treats all assets equally with 2x leverage, which:
- Overestimates crypto capacity (should be 1x)
- Underestimates stock capacity (should be 4x intraday)
- Ignores leverage fees (6.5% annual significant cost)
- Doesn't model EOD deleveraging (2x constraint)

### Required Changes
1. Separate crypto and stock capital pools
2. Apply 1x constraint to crypto, 4x to stocks
3. Model leverage fees for stocks
4. Test different portfolio mixes:
   - All crypto (tight capacity, no fees)
   - All stocks (loose capacity, high fees)
   - Mixed (realistic scenario)

### Work Stealing Priority
- **Crypto**: Very important (only 3-5 pairs, 1x limit, 2 orders = 50%+ portfolio)
- **Stocks**: Important (many symbols, 4x leverage, but still hit limits)

Crypto work stealing is CRITICAL because with only 3 cryptos and 1x leverage:
- 2 positions = 66% of crypto capital
- 3 positions = 100% of crypto capital
- Any 4th crypto MUST steal

Stocks have more breathing room with 4x leverage.
