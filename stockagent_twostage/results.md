# Two-Stage Daily Trading Backtest Results

## Configuration
- **Model**: Claude Opus 4.5 with Extended Thinking (60K token budget)
- **Symbols**: BTCUSD, ETHUSD, LINKUSD
- **Period**: 16 trading days (2025-09-30 to 2025-10-15)
- **Initial Capital**: $100,000
- **Min Confidence**: 0.3
- **Frequency**: Daily

## Strategy Overview
1. **Stage 1 (Portfolio Allocation)**: Opus 4.5 with ultrathink analyzes:
   - Historical price movements in basis points
   - Chronos2 neural forecasts
   - Outputs allocation weights (0-1) per symbol

2. **Stage 2 (Price Prediction)**: Claude Sonnet sets:
   - Entry price (bps from close)
   - Exit price (profit target)
   - Stop loss price

## Results

| Metric | Value |
|--------|-------|
| **Total Return** | **+10.57%** |
| Annualized Return | 1054.11% |
| Final Equity | $110,574.23 |
| Total Trades | 24 |
| Win Rate | 54.2% |
| Avg Confidence | 0.56 |

## Daily Performance

| Date | PnL | Return | Equity | Trades |
|------|-----|--------|--------|--------|
| 2025-09-30 | +$2,525.75 | +2.526% | $102,525.75 | 2 |
| 2025-10-01 | +$34.22 | +0.033% | $102,559.97 | 2 |
| 2025-10-02 | +$189.20 | +0.184% | $102,749.17 | 2 |
| 2025-10-03 | -$40.46 | -0.039% | $102,708.71 | 1 |
| 2025-10-04 | +$1,037.20 | +1.010% | $103,745.90 | 1 |
| 2025-10-05 | ... | ... | ... | ... |
| 2025-10-06 | ... | ... | ... | ... |
| 2025-10-07 | ... | ... | ... | ... |
| 2025-10-08 | ... | ... | ... | ... |
| 2025-10-09 | ... | ... | ... | ... |
| 2025-10-10 | -$484.55 | -0.454% | $106,333.93 | 1 |
| 2025-10-11 | +$1,987.69 | +1.869% | $108,321.61 | 1 |
| 2025-10-12 | +$3,394.03 | +3.133% | $111,715.64 | 2 |
| 2025-10-13 | -$1,193.64 | -1.068% | $110,522.00 | 2 |
| 2025-10-14 | +$52.23 | +0.047% | $110,574.23 | 2 |

## Key Observations
- **Strong performance**: +10.57% return over 16 trading days
- **Selective trading**: Only 24 trades with 54.2% win rate
- **Risk management**: Stop losses triggered on losing trades limited downside
- **Best days**: Oct 12 (+3.13%), Sep 30 (+2.53%), Oct 11 (+1.87%)
- **Worst days**: Oct 13 (-1.07%), Oct 10 (-0.45%)
- **Asset preference**: Model favored ETH and LINK over BTC (higher volatility)
- **Extended thinking**: 60K token budget provided detailed reasoning

## Run Command
```bash
python -m stockagent_twostage.backtest \
    --symbols BTCUSD ETHUSD LINKUSD \
    --days 10 \
    --thinking \
    --async \
    --parallel-days 5
```
