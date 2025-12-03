# Two-Stage Hourly Trading Backtest Results

## Configuration
- **Model**: Claude Opus 4.5 with Extended Thinking (60K token budget)
- **Symbols**: BTCUSD, ETHUSD, LINKUSD
- **Period**: 120 hours (5 days, 2024-10-27 to 2024-11-01)
- **Initial Capital**: $100,000
- **Min Confidence**: 0.3
- **Frequency**: Hourly
- **Data Source**: trainingdatahourly/crypto/

## Strategy Overview
1. **Stage 1 (Portfolio Allocation)**: Opus 4.5 with ultrathink analyzes:
   - Historical HOURLY price movements in basis points
   - Outputs allocation weights (0-1) per symbol
   - Adjusted thresholds: 20 bps allocation threshold (vs 50 bps daily)

2. **Stage 2 (Price Prediction)**: Claude Sonnet sets:
   - Entry price: -25 to -35 bps (tighter than daily)
   - Exit price: +75 to +85 bps (smaller targets)
   - Stop loss: -60 to -65 bps

## Hourly vs Daily Differences
| Parameter | Daily | Hourly |
|-----------|-------|--------|
| Allocation threshold | 50 bps | 20 bps |
| Mean reversion | 300 bps | 100 bps |
| Entry offset | -50 to -100 bps | -25 to -35 bps |
| Exit target | +150 to +400 bps | +75 to +85 bps |
| Leverage cost | 1.78 bps/day | 0.07 bps/hour |

## Results

| Metric | Value |
|--------|-------|
| **Total Return** | **-7.07%** |
| Annualized Return | -99.53% |
| Final Equity | $92,930.87 |
| Total Trades | 143 |
| Win Rate | 28.0% |
| Avg Confidence | 0.68 |
| Hours Tested | 120 |

## Key Observations
- **Negative performance**: -7.07% return over 120 hours (5 days)
- **High trade frequency**: 143 trades vs 24 for daily (6x more)
- **Low win rate**: Only 28% of trades profitable
- **Higher confidence**: Model was more confident (0.68 vs 0.56) but less accurate
- **Transaction costs**: More trades = more fees eating into returns
- **Volatility mismatch**: Hourly crypto volatility proved challenging for limit orders

## Comparison: Daily vs Hourly

| Metric | Daily | Hourly |
|--------|-------|--------|
| **Total Return** | **+10.57%** | -7.07% |
| Win Rate | **54.2%** | 28.0% |
| Total Trades | 24 | 143 |
| Avg Confidence | 0.56 | 0.68 |
| Time Period | 16 days | 5 days |
| Annualized | +1054% | -99.5% |

## Analysis

The hourly strategy significantly underperformed the daily strategy:

1. **Noise vs Signal**: Hourly data has more noise; the model was often triggered by false signals
2. **Entry execution**: Limit order entries (-25 to -35 bps) rarely hit during rapid moves
3. **Tight stops**: 60 bps stops were too tight for hourly crypto volatility
4. **Overconfidence**: Model confidence (0.68) didn't correlate with actual success
5. **Market period**: Oct 27-Nov 1 2024 was a challenging period with high volatility

## Recommendations

1. **Wider stops**: Consider 80-100 bps stops for hourly trading
2. **Market orders**: Use market orders instead of limit orders for entry
3. **Momentum filters**: Avoid mean-reversion in trending markets
4. **Reduce positions**: Smaller position sizes for hourly timeframe
5. **Different period**: Test on different market conditions

## Conclusion

**Daily trading with Opus 4.5 thinking is the clear winner** for this strategy:
- Daily: +10.57% return, 54% win rate
- Hourly: -7.07% return, 28% win rate

The additional granularity of hourly trading does not translate to better returns, and the increased transaction costs and noise significantly hurt performance.

## Run Command
```bash
python -m stockagent_twostage_hourly.backtest \
    --symbols BTCUSD ETHUSD LINKUSD \
    --hours 120 \
    --thinking \
    --async
```
