# Stock Trading Agents Comparison

## Overview

This document compares the different AI-powered stock trading agents in this codebase. All agents use LLM models to generate trading plans with limit-style entries and exits.

## Agent Strategies

### 1. stockagentdeepseek (Base Agent)
**Location:** `stockagentdeepseek/`

**Strategy:** Base DeepSeek trading agent that generates limit-style trading plans.

**Key Features:**
- Uses DeepSeek Reasoner model to analyze market data and produce trading plans
- Supports day-by-day replanning with portfolio state updates
- Includes risk strategies: ProbeTradeStrategy and ProfitShutdownStrategy
- Leverage guidance: 4× intraday gross exposure, reduced to 2× by close
- 6.75% annual interest on borrowed capital above available cash

**Benchmark Results:**
| Metric | Value |
|--------|-------|
| Realized PnL | $7.22 |
| Fees | $0.56 |
| Net PnL | $6.65 |
| Ending Equity | $8,006.94 |

---

### 2. stockagentdeepseek_entrytakeprofit
**Location:** `stockagentdeepseek_entrytakeprofit/`

**Strategy:** Entry/take-profit evaluation pipeline - focuses on entry price triggers and take-profit targets.

**Key Features:**
- Uses EntryTakeProfitSimulator for execution
- More conservative approach with strict limit orders
- Only executes when entry price is touched within day's range

**Benchmark Results:**
| Metric | Value |
|--------|-------|
| Realized PnL | $0.00 |
| Fees | $0.56 |
| Net PnL | -$0.56 |
| Daily Return | -0.70% |
| Annual Return | -176.02% |

**Analysis:** Poor performance in test scenario - strict entry limits often not filled.

---

### 3. stockagentdeepseek_maxdiff
**Location:** `stockagentdeepseek_maxdiff/`

**Strategy:** Max-diff execution pipeline - optimizes for maximum price differential capture.

**Key Features:**
- MaxDiffSimulator for limit-entry/exit execution
- Only trades when price triggers are touched
- Separate fee rates: 0.05% for stocks, 0.15% for crypto
- Targets high-low range exploitation

**Benchmark Results:**
| Metric | Value |
|--------|-------|
| Realized PnL | $0.00 |
| Fees | $0.00 |
| Net PnL | $0.00 |
| Daily Return | 0.00% |

**Analysis:** No fills in test scenario - strategy is opportunistic.

---

### 4. stockagentdeepseek_neural
**Location:** `stockagentdeepseek_neural/`

**Strategy:** Neural forecast integration - augments DeepSeek planning with ML price predictions.

**Key Features:**
- Integrates CombinedForecastGenerator (Toto, Kronos, Chronos2 models)
- Provides neural forecasts for OHLC prices to the LLM
- MAE-weighted model selection for forecast combination
- Same execution as base agent but with forecast context

**Benchmark Results:**
| Metric | Value |
|--------|-------|
| Realized PnL | $7.22 |
| Fees | $0.56 |
| Net PnL | $6.65 |
| Ending Equity | $8,006.94 |

**Analysis:** Same as base agent in this test - forecast context aids LLM reasoning.

---

### 5. stockagentdeepseek_combinedmaxdiff
**Location:** `stockagentdeepseek_combinedmaxdiff/`

**Strategy:** Neural DeepSeek planning with max-diff execution and signal calibration.

**Key Features:**
- Combines neural forecasts with max-diff execution
- Signal calibration using linear regression on historical predictions
- Separate annualization: 252 days/year for equities, 365 days/year for crypto
- Tracks calibration metrics: slope, intercept, raw/calibrated expected moves

**Calibration Metrics:**
- `{symbol}_calibration_slope`: Sensitivity of predictions to actual returns
- `{symbol}_calibration_intercept`: Bias correction
- `{symbol}_raw_expected_move_pct`: Model's raw forecast
- `{symbol}_calibrated_expected_move_pct`: Bias-corrected forecast

**Analysis:** Most sophisticated strategy - combines ML forecasts with calibrated execution.

---

## Performance Summary

| Agent | Net PnL | Daily Return | Strategy Type |
|-------|---------|--------------|---------------|
| **stockagentdeepseek** | $6.65 | +0.08% | Base LLM |
| **stockagentdeepseek_neural** | $6.65 | +0.08% | LLM + Neural |
| stockagentdeepseek_entrytakeprofit | -$0.56 | -0.70% | Conservative Limits |
| stockagentdeepseek_maxdiff | $0.00 | 0.00% | Opportunistic |
| stockagentdeepseek_combinedmaxdiff | Varies | Varies | Neural + Calibrated |

## Recommended Strategy Elements

Based on analysis, the best performing strategies combine:

1. **Neural Forecasts**: ML price predictions provide valuable context for LLM reasoning
2. **Signal Calibration**: Adjusting predictions based on historical accuracy
3. **Flexible Execution**: Not too strict on limit fills (base agent > entry_takeprofit)
4. **Risk Management**: Leverage limits and profit shutdown strategies
5. **Asset-Aware Annualization**: 252 days for stocks, 365 for crypto

## Production Trade History (Reference)

From `evaltests/baseline_pnl_summary.json`:
- Total trades: 68
- Total realized PnL: -$8,661.71
- Best symbol: BTCUSD (+$356.73)
- Worst symbol: MSFT (-$8,549.65)
- Duration: 7.1 days

*Note: Production loss was primarily due to a single large MSFT position.*

---

### 6. stockagentopus (Claude Opus 4.5 Agent) - NEW
**Location:** `stockagentopus/`

**Strategy:** Claude Opus 4.5-based trading agent combining the best elements from all previous strategies.

**Key Features:**
- Uses Claude Opus 4.5 (`claude-opus-4-5-20251101`) for advanced reasoning
- Neural forecast integration with CombinedForecastGenerator
- Signal calibration using linear regression on historical predictions
- Max-diff execution for limit-style trading
- On-disk caching for reproducibility and cost savings
- Extended thinking capabilities for complex market analysis
- Separate annualization: 252 days for equities, 365 days for crypto

**Architecture:**
```
stockagentopus/
├── __init__.py          # Package exports
├── agent.py             # Core simulation logic
├── opus_wrapper.py      # Claude Opus API wrapper with caching
├── prompt_builder.py    # Prompt construction utilities
└── backtest.py          # Backtest runner script
```

**API Usage:**
```python
from stockagentopus import simulate_opus_plan, simulate_opus_replanning

# Single day simulation
result = simulate_opus_plan(
    market_data=bundle,
    account_snapshot=snapshot,
    target_date=target_date,
    symbols=["AAPL", "MSFT"],
    calibration_window=14,
)

# Multi-day backtest
result = simulate_opus_replanning(
    market_data_by_date={date1: bundle1, date2: bundle2},
    account_snapshot=snapshot,
    target_dates=[date1, date2],
)

print(result.summary())
```

**CLI Backtest:**
```bash
# With real Anthropic API
python -m stockagentopus.backtest --symbols AAPL MSFT --days 5

# With mock LLM for testing
python -m stockagentopus.backtest --mock --symbols AAPL --days 3
```

**Caching:**
- In-memory cache (configurable TTL, default 1 hour)
- On-disk cache in `.opus_cache/` directory for reproducibility
- Cache key based on: model, messages, temperature, max_tokens

**Environment Variables:**
```bash
ANTHROPIC_API_KEY=sk-...          # Required for API calls
OPUS_MAX_OUTPUT_TOKENS=20000      # Max response tokens
OPUS_CONTEXT_LIMIT=200000         # Max context window
OPUS_TEMPERATURE=1.0              # Sampling temperature
OPUS_CACHE_DIR=.opus_cache        # Disk cache location
OPUS_MAX_ATTEMPTS=3               # API retry attempts
```

---

## Performance Summary

| Agent | Net PnL | Daily Return | Strategy Type |
|-------|---------|--------------|---------------|
| **stockagentdeepseek** | $6.65 | +0.08% | Base LLM |
| **stockagentdeepseek_neural** | $6.65 | +0.08% | LLM + Neural |
| **stockagentopus** | TBD | TBD | Opus + Neural + Calibrated |
| stockagentdeepseek_entrytakeprofit | -$0.56 | -0.70% | Conservative Limits |
| stockagentdeepseek_maxdiff | $0.00 | 0.00% | Opportunistic |
| stockagentdeepseek_combinedmaxdiff | Varies | Varies | Neural + Calibrated |

## Recommended Strategy Elements

Based on analysis, the best performing strategies combine:

1. **Neural Forecasts**: ML price predictions provide valuable context for LLM reasoning
2. **Signal Calibration**: Adjusting predictions based on historical accuracy
3. **Flexible Execution**: Not too strict on limit fills (base agent > entry_takeprofit)
4. **Risk Management**: Leverage limits and profit shutdown strategies
5. **Asset-Aware Annualization**: 252 days for stocks, 365 for crypto
6. **Advanced LLM**: Claude Opus 4.5 provides extended thinking for complex market analysis

## Production Trade History (Reference)

From `evaltests/baseline_pnl_summary.json`:
- Total trades: 68
- Total realized PnL: -$8,661.71
- Best symbol: BTCUSD (+$356.73)
- Worst symbol: MSFT (-$8,549.65)
- Duration: 7.1 days

*Note: Production loss was primarily due to a single large MSFT position.*

---

## Running Real Backtests

To run a real backtest with the OpusAgent:

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# Run backtest
python -m stockagentopus.backtest --symbols AAPL MSFT GOOG --days 10 --capital 100000

# Save results to file
python -m stockagentopus.backtest --symbols AAPL --days 5 --output results.json
```

## Development

To run tests:
```bash
pytest tests/prod/agents/stockagentopus/ -v
```
