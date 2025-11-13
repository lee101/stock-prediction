# Risk Management & Position Sizing Test Plan

## Overview
Test and validate risk management and position sizing strategies using precomputed trade data from strategytraining/.

## Test Objectives

### 1. **Validate Current Production Sizing**
   - Test Kelly_50pct @ 4x leverage performance
   - Compare against baseline strategies
   - Identify edge cases where it underperforms

### 2. **Risk Limit Testing**
   - Verify MAX_SYMBOL_EXPOSURE_PCT (60%) enforcement
   - Test MAX_TOTAL_EXPOSURE_PCT (120%) compliance
   - Validate leverage limits (4x intraday, 2x overnight)
   - Check crypto constraints (no leverage, no shorting)

### 3. **Strategy Comparison**
   - Compare all sizing strategies on same dataset
   - Measure risk-adjusted returns (Sharpe ratio)
   - Analyze drawdown characteristics
   - Evaluate position sizing stability

### 4. **Edge Case Testing**
   - High volatility periods (crypto crashes, market selloffs)
   - Low volatility grinding markets
   - Correlation breakdowns (all positions correlated)
   - Position concentration scenarios

## Test Suite

### Test 1: Current Production Simulation
**File**: `test_current_production_sizing.py`
**Goal**: Simulate exactly what's running in production

```python
# Test Kelly_50pct @ 4x with actual exposure limits
# Compare to baseline 50% fixed allocation
# Measure: Sharpe, max DD, exposure violations
```

### Test 2: Risk Limit Validation
**File**: `test_risk_limits.py`
**Goal**: Verify all risk limits are enforced correctly

```python
# Test scenarios:
# - Multiple positions reaching 60% symbol exposure
# - Total portfolio approaching 120% exposure
# - Leverage limits during volatile moves
# - Crypto-specific constraints
```

### Test 3: Strategy Benchmarking
**File**: `strategytraining/test_sizing_on_precomputed_pnl.py` (EXISTS)
**Goal**: Compare all strategies on historical trades

Current results show:
- VolAdjusted strategies: Best Sharpe (2.0+)
- Kelly strategies: Highest returns (49%)
- Fixed strategies: Most stable

### Test 4: Volatility Regime Testing
**File**: `test_volatility_regimes.py`
**Goal**: Test strategies across different market conditions

```python
# Split data by volatility regime:
# - Low vol (VIX < 15): Test aggressive sizing
# - Medium vol (VIX 15-25): Test balanced sizing
# - High vol (VIX > 25): Test defensive sizing
```

### Test 5: Correlation Risk Testing
**File**: `test_correlation_risk.py`
**Goal**: Validate correlation-aware strategies

```python
# Test CorrelationAwareStrategy in:
# - Diversified portfolio (low correlation)
# - Correlated portfolio (2008 crash, COVID)
# - Single-asset vs multi-asset sizing
```

### Test 6: Drawdown Management
**File**: `test_drawdown_limits.py`
**Goal**: Test max drawdown limits and recovery

```python
# Simulate LIVE_DRAWDOWN_TRIGGER behavior
# Test probe trade transitions
# Validate position reduction during drawdowns
```

## Metrics to Track

### Performance Metrics
- Total PnL
- Annualized return
- Sharpe ratio
- Sortino ratio
- Win rate

### Risk Metrics
- Maximum drawdown
- Average drawdown
- Drawdown duration
- Value at Risk (VaR 95%)
- Conditional VaR (CVaR)

### Sizing Metrics
- Average position size
- Position size volatility
- Leverage utilization
- Exposure limit violations
- Turnover rate

### Execution Metrics
- Slippage impact
- Fee impact
- Number of trades
- Average trade duration

## Expected Outcomes

### Hypothesis 1: VolAdjusted strategies will outperform Kelly @ 4x
**Rationale**: Recent tests show 2.12 Sharpe for VolAdjusted vs ~1.0 for Kelly variants
**Test**: Compare Kelly_50pct @ 4x vs VolAdjusted_15pct on production data

### Hypothesis 2: 4x leverage increases returns but degrades Sharpe
**Rationale**: Higher leverage amplifies both gains and losses
**Test**: Compare Kelly_50pct @ 1x, 2x, 4x leverage

### Hypothesis 3: Correlation-aware sizing reduces drawdowns
**Rationale**: Portfolio diversification reduces correlated losses
**Test**: Compare CorrAware vs Fixed strategies during high-correlation periods

### Hypothesis 4: Dynamic sizing outperforms fixed sizing
**Rationale**: Market conditions change, fixed sizing can't adapt
**Test**: Compare Fixed_50pct vs VolatilityAdjusted vs Kelly strategies

## Implementation Priority

### Phase 1: Validation (Week 1)
1. ✅ Run existing test_sizing_on_precomputed_pnl.py
2. Create test_current_production_sizing.py
3. Validate risk limits with test_risk_limits.py

### Phase 2: Analysis (Week 2)
4. Implement test_volatility_regimes.py
5. Run comprehensive strategy comparison
6. Analyze results and document findings

### Phase 3: Optimization (Week 3)
7. Identify best-performing strategy
8. Test with different parameters
9. Validate on holdout data

### Phase 4: Production Integration (Week 4)
10. Update src/sizing_utils.py with best strategy
11. Add monitoring and alerting
12. Deploy to paper trading
13. Validate for 1 week before live

## Success Criteria

### Must Have
- [ ] All risk limits enforced (no violations)
- [ ] Better Sharpe ratio than current production (1.42 baseline)
- [ ] Max drawdown < 15%
- [ ] Win rate > 55%

### Should Have
- [ ] Sharpe ratio > 2.0 (match VolAdjusted strategies)
- [ ] Max drawdown < 12%
- [ ] Total return > 25% (annualized)
- [ ] Low turnover (< 100% per year)

### Nice to Have
- [ ] Sharpe ratio > 2.5
- [ ] Consistent performance across all symbols
- [ ] Low correlation to market indices
- [ ] Graceful degradation during black swan events

## Notes

- All tests should use strategytraining/datasets/full_strategy_dataset_*_trades.parquet
- Results should be saved to strategytraining/reports/
- Document all assumptions and limitations
- Track computational cost of each strategy
