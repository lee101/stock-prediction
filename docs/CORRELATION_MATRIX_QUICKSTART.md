# Correlation Matrix Quickstart

Built: 2025-11-13

## What Is This?

A correlation matrix computed from your training data that shows how different assets move together. This helps manage portfolio risk by identifying concentrated positions in correlated assets.

## Files Created

```
trainingdata/
  ├── correlation_matrix.pkl          # Binary format (fast loading)
  ├── correlation_matrix.json         # JSON format (human-readable)
  ├── correlation_matrix.yaml         # YAML format (also human-readable)
  ├── correlation_matrix.csv          # CSV format (for spreadsheets)
  ├── volatility_metrics.csv          # Volatility metrics CSV
  ├── correlation_matrix_20251113.pkl # Dated backup
  ├── build_correlation_matrix_from_csvs.py  # Builder script
  └── load_correlation_utils.py       # Loader utilities

examples/
  ├── use_correlation_matrix.py       # Usage examples
  └── risk_adjusted_sizing.py         # Risk-adjusted position sizing
```

## Quick Usage

### 1. Rebuild the Matrix (when training data changes)

```bash
source .venv/bin/activate
python trainingdata/build_correlation_matrix_from_csvs.py --lookback 250
```

Options:
- `--lookback 250` - Use last 250 days (default: all data)
- `--threshold 0.7` - Correlation threshold for clustering (default: 0.7)
- `--input trainingdata/train` - Input directory (default: trainingdata/train)
- `--output trainingdata` - Output directory (default: trainingdata)

### 2. Load and Use in Python

```python
from trainingdata.load_correlation_utils import (
    load_correlation_matrix,
    get_correlation,
    get_symbol_correlations,
    get_portfolio_diversification_metrics,
    get_volatility_metrics,
    get_top_volatile_symbols,
    get_best_sharpe_symbols,
)

# Load the matrix
corr_data = load_correlation_matrix()

# Get correlation between two symbols
corr = get_correlation(corr_data, 'BTCUSD', 'ETHUSD')
print(f"BTC-ETH correlation: {corr:.3f}")

# Get top correlated symbols
corrs = get_symbol_correlations(corr_data, 'AAPL', top_n=10)

# Check portfolio diversification
portfolio = {'AAPL': 50000, 'MSFT': 30000, 'BTCUSD': 20000}
metrics = get_portfolio_diversification_metrics(corr_data, portfolio)
print(f"Effective number of bets: {metrics['effective_num_bets']:.2f}")

# Get volatility metrics
vol_metrics = get_volatility_metrics(corr_data, 'BTCUSD')
print(f"BTCUSD volatility: {vol_metrics['annualized_volatility']:.1%}")
print(f"Sharpe ratio: {vol_metrics['sharpe_ratio']:.2f}")

# Find best risk-adjusted opportunities
best_sharpe = get_best_sharpe_symbols(corr_data, n=5)
print(best_sharpe)
```

### 3. Run Examples

```bash
# Basic usage examples
python examples/use_correlation_matrix.py

# Risk-adjusted position sizing
python examples/risk_adjusted_sizing.py
```

## Data Structure

The correlation matrix file contains:

```python
{
    "timestamp": "2025-11-13T04:38:32+00:00",
    "lookback_days": 250,
    "symbols": ["AAPL", "MSFT", ...],
    "correlation_matrix": [[1.0, 0.85, ...], ...],  # NxN matrix
    "data_quality": {
        "AAPL": {"valid_periods": 250, "missing_periods": 0, "data_pct": 100.0},
        ...
    },
    "clusters": {
        "cluster_1": {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "size": 3,
            "avg_correlation": 0.82,
            "label": "Tech Stocks"
        },
        ...
    },
    "volatility_metrics": {
        "AAPL": {
            "annualized_volatility": 0.311,
            "rolling_vol_30d": 0.285,
            "rolling_vol_60d": 0.298,
            "downside_volatility": 0.245,
            "max_drawdown": -0.525,
            "var_95": -0.0356,
            "var_99": -0.0512,
            "sharpe_ratio": -2.27,
            "sortino_ratio": -2.85,
            "skewness": 0.12,
            "kurtosis": 3.45,
            "mean_return_annualized": -0.706,
            "volatility_percentile": 42.3
        },
        ...
    },
    "metadata": {
        "num_symbols": 136,
        "num_clusters": 13,
        "avg_correlation": 0.65,
        "avg_volatility": 0.408,
        "avg_sharpe": 0.484
    }
}
```

## Identified Clusters (Latest)

Based on the most recent run with 250-day lookback:

1. **Mixed Group** (19 symbols, avg corr=0.774)
   - Major indices and financials: SPY, QQQ, IWM, AXP, GS, BAC, etc.

2. **Crypto Assets** (18 symbols, avg corr=0.759)
   - BTC, ETH, SOL, AVAX, LINK, UNI, DOGE, etc.

3. **Energy** (6 symbols, avg corr=0.829)
   - XOM, CVX, COP, SLB, EOG, XLE

4. **REITs** (3 symbols, avg corr=0.745)
   - REIT, PSA, PLD

5. **International ETFs** (3 symbols, avg corr=0.910)
   - EFA, EEM, VXUS

Plus 8 smaller clusters of 2-3 highly correlated symbols.

## Key Metrics

### Correlation Metrics
- **Average Correlation**: Used to assess overall market correlation
- **Effective Number of Bets (ENB)**: Portfolio diversification measure
  - ENB < 3: Highly concentrated
  - ENB 3-5: Moderate diversification
  - ENB > 5: Well diversified
- **Concentration Score**: 1/ENB (lower is better)

### Volatility Metrics
- **Annualized Volatility**: Standard deviation of returns (annualized)
- **Rolling Volatility**: 30-day and 60-day rolling volatility
- **Downside Volatility**: Volatility of negative returns only
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR (95%, 99%)**: Value at Risk at 95th and 99th percentiles
- **Sharpe Ratio**: Return per unit of total risk (higher is better)
- **Sortino Ratio**: Return per unit of downside risk (higher is better)
- **Skewness**: Distribution asymmetry (positive = right tail)
- **Kurtosis**: Distribution tail heaviness (>3 = fat tails)
- **Volatility Percentile**: Current vol vs historical (>80 = elevated)

## Use Cases

### 1. Pre-Trade Risk Check

Before entering a position, check:
- What cluster does this symbol belong to?
- What's my current exposure to that cluster?
- Would this trade exceed cluster limits?
- What's the volatility and risk profile?

```python
# Correlation check
cluster = get_cluster_for_symbol(corr_data, 'SOLUSD')
current_exposure = get_cluster_exposure(corr_data, positions, cluster['cluster_id'])

# Volatility check
vol_metrics = get_volatility_metrics(corr_data, 'SOLUSD')
print(f"Volatility: {vol_metrics['annualized_volatility']:.1%}")
print(f"Sharpe: {vol_metrics['sharpe_ratio']:.2f}")
```

### 2. Portfolio Monitoring

Daily check:
- Current portfolio diversification (ENB)
- Average pairwise correlation
- Cluster exposures

```python
metrics = get_portfolio_diversification_metrics(corr_data, positions)
if metrics['effective_num_bets'] < 3:
    print("WARNING: Portfolio is too concentrated!")
```

### 3. Risk-Adjusted Position Sizing

Adjust position sizes based on both correlation and volatility:
- High volatility: smaller positions
- Low volatility: larger positions
- Diversifying (low correlation): can be larger
- Concentrating (high correlation): should be smaller

```python
# Volatility-adjusted sizing
target_vol = 0.15  # 15% target contribution
symbol_vol = get_volatility_metrics(corr_data, symbol)['annualized_volatility']
vol_scalar = target_vol / symbol_vol
adjusted_size = base_size * vol_scalar
```

## Data Quality Notes

From the latest run:
- **136 symbols** analyzed
- **Data coverage**: 6.7% (due to 250-day lookback with varying data availability)
- **NaN correlations**: Some symbol pairs have NaN correlations due to non-overlapping time periods

To improve coverage:
1. Use shorter lookback: `--lookback 60`
2. Use all available data: omit `--lookback` parameter
3. Ensure training data has sufficient overlap

## Updating Schedule

Recommended:
- **Daily**: For live trading (catches regime changes)
- **Weekly**: For backtesting (reduces computation)
- **After data refresh**: When new training data is downloaded

## Integration with Trading System

See `docs/correlation_risk_management_plan.md` for full integration plan:

### Phase 1: Monitoring (Current)
- Calculate and save correlation matrix
- Log diversification metrics
- Generate reports

### Phase 2: Soft Limits (TODO)
- Warning when cluster limits approached
- Log correlation violations

### Phase 3: Hard Limits (TODO)
- Enforce cluster exposure limits in `sizing_utils.py`
- Adjust position sizes based on correlation

### Phase 4: Optimization (Future)
- Correlation-aware rebalancing
- Risk budgeting across clusters
- Pairs trading opportunities

## Troubleshooting

### "Symbol not found in correlation matrix"
- Symbol may not be in training data
- Rebuild matrix with updated training data

### "NaN correlations"
- Insufficient overlapping data
- Try shorter lookback or check data quality

### "Cannot join tz-naive with tz-aware DatetimeIndex"
- Already fixed in `build_correlation_matrix_from_csvs.py`
- Uses `utc=True` for timestamp parsing

## Performance

- **Computation time**: ~3 seconds for 136 symbols
- **File sizes**:
  - PKL: 170KB (fastest to load)
  - JSON: 385KB (human-readable)
  - YAML: 331KB (also human-readable)
  - CSV (correlation): 218KB (matrix only, no metadata)
  - CSV (volatility): ~15KB (metrics table)

## Latest Results (2025-11-13)

### Key Statistics
- **136 symbols** analyzed
- **13 correlation clusters** identified
- **Average volatility**: 40.8% annualized
- **Average Sharpe ratio**: 0.48

### Top Volatile Assets
1. ADA-USD: 106.1% annualized vol
2. XLM-USD: 101.0% annualized vol
3. ALGO-USD: 98.6% annualized vol

### Best Sharpe Ratios
1. PLTR: 2.63
2. SAP: 2.52
3. GLD: 2.09

### Main Clusters
1. Mixed Group (19 symbols): Financials + major indices
2. Crypto Assets (18 symbols): BTC, ETH, altcoins
3. Energy (6 symbols): Oil & gas stocks

## References

- Implementation: `trainingdata/build_correlation_matrix_from_csvs.py`
- Utilities: `trainingdata/load_correlation_utils.py`
- Examples:
  - `examples/use_correlation_matrix.py`
  - `examples/risk_adjusted_sizing.py`
- Full plan: `docs/correlation_risk_management_plan.md`
