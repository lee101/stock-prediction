# Position Sizing Strategy Experiments

## Overview

This framework tests different position sizing strategies to maximize profit while managing risk. It accounts for:
- **2x max leverage** for stocks/ETFs
- **6.75% annual interest** (calculated daily) on leveraged positions
- **Crypto constraints**: no leverage, no shorting
- Model predictions from compiled Kronos/Toto models

## Quick Start

### Run Simple Tests (No Data Required)
```bash
python marketsimulator/test_sizing_strategies_simple.py
```

### Run Quick Experiment (2 symbols, 3 strategies)
```bash
export MARKETSIM_FAST_SIMULATE=1
./marketsimulator/run_sizing_experiments.sh quick
```

### Run Full Experiment (All symbols and strategies)
```bash
export MARKETSIM_FAST_SIMULATE=1
./marketsimulator/run_sizing_experiments.sh full
```

### Run Custom Experiment
```bash
export MARKETSIM_FAST_SIMULATE=1
python marketsimulator/test_sizing_strategies.py \
    --symbols BTCUSD ETHUSD AAPL MSFT \
    --strategies kelly_25 optimal_f fixed_50 \
    --output-dir marketsimulator/results/custom
```

## Sizing Strategies

### Fixed Fraction Strategies
- `fixed_25`: Always allocate 25% of equity
- `fixed_50`: Always allocate 50% of equity
- `fixed_75`: Always allocate 75% of equity
- `fixed_100`: Always allocate 100% of equity (full Kelly allowed up to 2x leverage)

### Kelly-Based Strategies
Optimal allocation based on edge/variance ratio with fractional scaling:
- `kelly_10`: 10% of full Kelly (very conservative)
- `kelly_25`: 25% of full Kelly (conservative, recommended)
- `kelly_50`: 50% of full Kelly (aggressive)

Formula: `f = fraction * (expected_return / variance)`

### Volatility Targeting
Size positions to achieve target portfolio volatility:
- `voltarget_10`: Target 10% annual volatility
- `voltarget_15`: Target 15% annual volatility

Formula: `position_size = target_vol / asset_vol`

### Risk Parity
Equal risk contribution across positions:
- `riskparity_5`: 5% risk per position
- `riskparity_10`: 10% risk per position

Formula: `position_size = target_risk / asset_vol`

### Optimal F (Leverage-Cost Adjusted)
Maximizes expected log growth accounting for leverage costs:
- `optimal_f`: Adjusts Kelly formula to subtract leverage cost when using >1x

Formula:
```
If position <= 1.0:
    f = expected_return / variance

If position > 1.0:
    f = (expected_return - daily_leverage_cost) / variance
```

This is the most theoretically sound approach for your constraints.

## Key Constraints

### Leverage Limits
- **Stocks/ETFs**: Max 2x leverage
- **Crypto**: Max 1x (no leverage)
- Positions are automatically capped at max leverage

### Leverage Costs
- **Annual rate**: 6.75% (configurable via `LEVERAGE_COST_ANNUAL`)
- **Daily cost**: 6.75% / 252 = 0.0268% per day
- **Applied to**: Leveraged portion only (exposure > equity)
- **Crypto**: No leverage costs (can't leverage)

Example:
- Equity: $100k
- Position: $150k (1.5x leverage)
- Daily cost: ($150k - $100k) × 0.0268% = $13.39/day

### Shorting
- **Stocks/ETFs**: Shorting allowed (negative positions)
- **Crypto**: No shorting (position forced to 0 if prediction negative)

## Output Files

### Summary CSV
`marketsimulator/results/sizing_strategy_results_TIMESTAMP.csv`

Columns:
- `strategy`: Strategy name
- `symbol`: Trading symbol
- `total_return`: Overall return (%)
- `sharpe_ratio`: Annualized Sharpe ratio
- `max_drawdown`: Maximum drawdown (%)
- `num_trades`: Number of trades executed
- `avg_leverage`: Average leverage used
- `leverage_cost`: Total leverage costs paid ($)
- `win_rate`: Percentage of winning trades
- `final_equity`: Final portfolio value ($)

### Detailed JSON
`marketsimulator/results/sizing_strategy_detailed_TIMESTAMP.json`

Contains:
- Full equity curves
- Individual trade records
- Timestamps for all events
- Complete performance metrics

## Environment Variables

### Speed Optimization
- `MARKETSIM_FAST_SIMULATE=1`: Reduce simulations (2x speedup)
- Uses 35 simulations instead of 50
- Auto-enables torch.compile + bf16 for 4-6x total speedup

### Leverage Configuration
- `LEVERAGE_COST_ANNUAL`: Annual leverage cost (default: 0.0675)
- `LEVERAGE_TRADING_DAYS`: Trading days per year (default: 252)
- `GLOBAL_MAX_GROSS_LEVERAGE`: Max leverage multiplier (default: 2.0)

### Model Configuration
- `TOTO_COMPILE=1`: Enable torch.compile for models
- `TOTO_DTYPE=bfloat16`: Use bf16 precision
- `KRONOS_DTYPE=bfloat16`: Use bf16 precision

## Expected Results

Based on theory and the Kelly criterion background you provided:

### Best Overall: `optimal_f` or `kelly_25`
- Accounts for leverage costs in sizing decisions
- Provides good risk-adjusted returns
- Avoids over-leveraging in volatile periods

### Most Aggressive: `riskparity_10` or `kelly_50`
- Highest potential returns
- Highest drawdowns
- May hit 2x leverage frequently
- High leverage costs

### Most Conservative: `kelly_10` or `fixed_25`
- Lower returns but much lower drawdowns
- Minimal leverage usage
- Good for risk-averse or uncertain predictions

### Volatility Targeting: `voltarget_10`
- Consistent risk exposure
- Good for portfolio allocation
- May underperform in low-vol regimes

## Analysis Tips

### 1. Sort by Sharpe Ratio
Best risk-adjusted performance:
```bash
sort -t',' -k4 -rn marketsimulator/results/sizing_strategy_results_*.csv | head -10
```

### 2. Check Leverage Costs
High leverage costs indicate strategy is using >1x frequently:
```bash
sort -t',' -k8 -rn marketsimulator/results/sizing_strategy_results_*.csv | head -10
```

### 3. Compare by Symbol
See which strategies work best for specific assets:
```bash
grep "BTCUSD" marketsimulator/results/sizing_strategy_results_*.csv | sort -t',' -k4 -rn
```

### 4. Evaluate Drawdowns
Find strategies with acceptable risk:
```bash
awk -F',' '$5 < 0.20' marketsimulator/results/sizing_strategy_results_*.csv
```

## Next Steps

### 1. Run Initial Experiment
```bash
export MARKETSIM_FAST_SIMULATE=1
./marketsimulator/run_sizing_experiments.sh quick
```

### 2. Analyze Results
Review the generated CSV to identify top performers.

### 3. Run Full Backtest
Once you identify promising strategies, run full backtests without FAST_SIMULATE:
```bash
unset MARKETSIM_FAST_SIMULATE
python marketsimulator/test_sizing_strategies.py \
    --symbols BTCUSD ETHUSD AAPL MSFT NVDA \
    --strategies optimal_f kelly_25 \
    --output-dir marketsimulator/results/final
```

### 4. Integrate with trade_stock_e2e.py
Replace the current `get_qty()` logic with your chosen strategy:
```python
from marketsimulator.sizing_strategies import SIZING_STRATEGIES, MarketContext

strategy = SIZING_STRATEGIES['optimal_f']
ctx = MarketContext(
    symbol=symbol,
    predicted_return=predicted_return,
    predicted_volatility=predicted_volatility,
    current_price=price,
    equity=alpaca_wrapper.equity,
    is_crypto=symbol in crypto_symbols,
)
result = strategy.calculate_size(ctx)
qty = result.quantity
```

## Files Created

- `marketsimulator/sizing_strategies.py`: Strategy implementations
- `marketsimulator/test_sizing_strategies.py`: Main experiment runner
- `marketsimulator/test_sizing_strategies_simple.py`: Quick validation tests
- `marketsimulator/run_sizing_experiments.sh`: Batch experiment script
- `docs/POSITION_SIZING_EXPERIMENTS.md`: This documentation

## Theory Notes (from your Kelly sizing background)

Your comprehensive Kelly discussion is excellent and the implementation follows those principles:

1. **Kelly is log-utility maximization**: Our `optimal_f` strategy directly maximizes E[log(1 + f*R - cost)]

2. **Shrinkage is critical**: When using real predictions, apply strong shrinkage to predicted returns (especially with short data windows like 7 days)

3. **Fractional Kelly is essential**: Full Kelly is too aggressive - the `kelly_25` strategy implements 0.25× Kelly as you suggested

4. **Drawdown caps**: The simulation tracks max drawdown and you can filter results by this metric

5. **Leverage costs matter**: The `optimal_f` strategy explicitly accounts for the 6.75% annual rate

6. **Multi-asset**: Each strategy can be applied independently per symbol, and the framework can be extended to consider correlations

The next evolution would be to:
- Add correlation-aware multi-asset Kelly
- Implement dynamic fractional Kelly based on prediction uncertainty
- Add Bayesian shrinkage to predictions
- Implement drawdown-based throttling
