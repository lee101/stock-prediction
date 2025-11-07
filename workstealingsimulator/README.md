# Work Stealing Parameter Optimizer

Simulates work stealing behavior using hourly crypto data and scipy optimization to find optimal configuration values.

## Data Sources

- **Hourly crypto data**: `../trainingdatahourly/` (BTCUSD, ETHUSD, etc.)
- **Strategy performance**: `full_strategy_dataset_20251101_211202_strategy_performance.parquet`
- **Trade history**: `full_strategy_dataset_20251101_211202_trades.parquet`

## Parameters Optimized

1. `crypto_ooh_force_count`: Top N cryptos to force immediate entry out-of-hours
2. `crypto_ooh_tolerance_pct`: Tolerance for non-forced cryptos out-of-hours
3. `crypto_normal_tolerance_pct`: Normal hours crypto tolerance
4. `entry_tolerance_pct`: Work stealing entry tolerance
5. `protection_pct`: Distance threshold to protect orders from stealing
6. `cooldown_seconds`: Cooldown after successful steal
7. `fight_threshold`: Steals before fighting detected
8. `fight_cooldown_seconds`: Fighting cooldown period

## Simulation Logic

1. Load hourly OHLCV data for all cryptos
2. For each hour, rank cryptos by forecasted PnL from strategy data
3. Attempt entries based on distance tolerance and market hours
4. Simulate work stealing when capacity blocked
5. Simulate exits at next day's high (simplified take-profit)
6. Track: total PnL, Sharpe ratio, win rate, steals, blocks

## Optimization Method

Uses scipy `differential_evolution`:
- Population-based global optimizer
- Robust to local minima
- Score = 0.5*total_pnl + 1000*sharpe + 2000*win_rate
- 20 iterations with population size 5

## Running

```bash
cd workstealingsimulator
python3 simulator.py
```

## Output

Prints optimal config values and final performance metrics:
- Total PnL across simulation period
- Sharpe ratio (annualized)
- Win rate
- Number of trades executed
- Number of successful steals
- Number of blocked orders
