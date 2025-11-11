# Quick Start: Position Sizing Experiments

## 1. Run Simple Validation (No Data Needed)

```bash
python marketsimulator/test_sizing_strategies_simple.py
```

This tests all 12 sizing strategies with synthetic data. Should complete in <10 seconds.

## 2. Run Quick Experiment (2 symbols, 5 strategies)

```bash
export MARKETSIM_FAST_SIMULATE=1
./marketsimulator/run_sizing_experiments.sh quick
```

This tests core strategies (fixed_50, kelly_25, optimal_f) on BTCUSD and AAPL.

Expected runtime: 2-5 minutes

## 3. View Results

```bash
# Find latest results
ls -lth marketsimulator/results/

# View summary
cat marketsimulator/results/experiment_*/sizing_strategy_results_*.csv

# Best by Sharpe ratio
head -1 marketsimulator/results/experiment_*/sizing_strategy_results_*.csv
tail -n +2 marketsimulator/results/experiment_*/sizing_strategy_results_*.csv | sort -t',' -k4 -rn | head -5
```

## 4. Run Full Experiment (All Symbols & Strategies)

```bash
export MARKETSIM_FAST_SIMULATE=1
./marketsimulator/run_sizing_experiments.sh full
```

This tests all 12 strategies on crypto, stocks, and ETFs.

Expected runtime: 10-30 minutes (depending on data availability)

## Key Files

- **Strategies**: `marketsimulator/sizing_strategies.py`
- **Runner**: `marketsimulator/test_sizing_strategies.py`
- **Results**: `marketsimulator/results/`
- **Docs**: `docs/POSITION_SIZING_EXPERIMENTS.md`

## What Gets Tested

### Strategies (12 total)
1. **Fixed %**: 25%, 50%, 75%, 100%
2. **Kelly**: 10%, 25%, 50% of full Kelly
3. **Vol Target**: 10%, 15% annual volatility
4. **Risk Parity**: 5%, 10% risk per position
5. **Optimal F**: Leverage-cost adjusted Kelly

### Constraints Enforced
- Stocks: Max 2x leverage
- Crypto: No leverage, no shorting
- Leverage cost: 6.75% annual on leveraged portion
- Realistic trade sizing

### Output Metrics
- Total return (%)
- Sharpe ratio (annualized)
- Max drawdown (%)
- Number of trades
- Average leverage
- Total leverage costs
- Win rate

## Expected Top Performers

Based on theory:
1. **optimal_f**: Best risk-adjusted returns with leverage cost awareness
2. **kelly_25**: Conservative Kelly, good Sharpe ratio
3. **voltarget_10**: Consistent risk, good for diversification

## Troubleshooting

### No price data found
If you see "Could not load price data", you need historical data. Options:
1. Use `marketsimulator/data_feed.py` to load existing data
2. Download new data to `strategytraining/raw_data/`
3. Use the fallback synthetic predictions (less accurate but works)

### Models not found
The script will fall back to synthetic predictions based on momentum/volatility if Kronos/Toto models aren't available. This is sufficient for comparing sizing strategies.

### Memory issues
Reduce the number of symbols or use FAST_SIMULATE mode.

## Integration with Live Trading

Once you identify the best strategy:

```python
# In trade_stock_e2e.py, replace get_qty() with:
from marketsimulator.sizing_strategies import SIZING_STRATEGIES, MarketContext

strategy = SIZING_STRATEGIES['optimal_f']  # or your chosen strategy

ctx = MarketContext(
    symbol=symbol,
    predicted_return=forecast_mean,
    predicted_volatility=forecast_std,
    current_price=current_price,
    equity=alpaca_wrapper.equity,
    is_crypto=symbol in crypto_symbols,
)

result = strategy.calculate_size(ctx)
qty = result.quantity
```

## Next Steps

1. Run `./marketsimulator/run_sizing_experiments.sh quick`
2. Review results in CSV
3. Pick top 2-3 strategies by Sharpe ratio
4. Run full backtest on those strategies without FAST_SIMULATE
5. Integrate winner into trade_stock_e2e.py
