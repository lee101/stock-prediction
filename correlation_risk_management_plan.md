# Correlation Risk Management Plan

**Date:** 2025-11-13
**Status:** Planning
**Priority:** P1 (High - addresses portfolio concentration risk)

---

## Problem Statement

Current position sizing allows up to 60% equity exposure per symbol. However, this doesn't account for correlation between positions, allowing concentrated risk through correlated positions:

**Example Risk Scenario:**
- 60% in AAPL (tech)
- 60% in MSFT (tech)
- 60% in GOOGL (tech)
- = 180% in highly correlated tech stocks

If tech sector drops 10%, portfolio could lose 18% despite "diversification" across symbols.

---

## Objectives

1. **Measure** correlation between all tradeable assets
2. **Limit** total exposure to correlated groups of assets
3. **Preserve** existing per-symbol limits (60% max)
4. **Improve** risk-adjusted returns through better diversification

---

## Proposed Solution: 3-Tier Risk Management

### Tier 1: Per-Symbol Limits (Existing)
- Max 60% equity per symbol
- Implemented in `src/sizing_utils.py`
- **Keep as-is** ✓

### Tier 2: Correlation Group Limits (NEW)
- Cluster symbols into correlation groups
- Limit total exposure per group
- Prevents concentrated sector/style bets

### Tier 3: Portfolio-Level Limits (NEW)
- Calculate portfolio-level effective exposure accounting for correlations
- Use as risk multiplier or constraint

---

## Correlation Matrix Implementation

### Data Collection
**Script:** `trainingdata/calculate_correlation_matrix.py`

**Requirements:**
- Historical price data for all tradeable symbols
- Rolling window: 60 trading days (3 months)
- Update frequency: Daily
- Storage: Pickle/JSON matrix + metadata

**Inputs:**
- List of all tradeable symbols from `alpaca_wrapper.py`
- Historical daily returns from existing data sources

**Outputs:**
- Correlation matrix (N x N where N = number of symbols)
- Timestamp of calculation
- Data quality metrics (% missing data per symbol)

### Clustering Approach

**Method 1: Hierarchical Clustering (Recommended)**
```
1. Calculate pairwise correlations
2. Convert correlation to distance: distance = sqrt(2 * (1 - correlation))
3. Apply hierarchical clustering with linkage threshold
4. Identify clusters with correlation > 0.7
```

**Advantages:**
- No need to specify number of clusters upfront
- Produces dendrogram for visualization
- Natural interpretation (highly correlated = same cluster)

**Method 2: Sector/Industry Tags**
- Use existing sector classifications
- Faster but less dynamic
- Doesn't capture style factors (momentum, value, etc.)

**Recommendation:** Use Method 1 (hierarchical) with Method 2 as fallback/validation

---

## Integration Points

### Phase 1: Monitoring & Alerts (Week 1-2)
**Goal:** Build awareness without changing trading behavior

**Implementation:**
1. Calculate correlation matrix daily
2. Compute current portfolio correlation metrics:
   - Average pairwise correlation of active positions
   - Max correlation between any two positions
   - Effective number of independent bets (ENB)
3. Log metrics to `trade_stock_e2e.log`
4. Generate daily correlation report

**New Metrics:**
```python
# Effective Number of Independent Bets
ENB = portfolio_value^2 / sum(position_i^2 * position_j^2 * correlation_ij)

# Portfolio concentration score (0-1, lower is more diversified)
concentration = 1 / ENB

# Correlation-adjusted exposure
effective_exposure = sqrt(sum(w_i * w_j * correlation_ij))
```

**Deliverables:**
- Daily correlation matrix file: `trainingdata/correlation_matrix_{date}.pkl`
- Daily report: `logs/correlation_report_{date}.txt`

### Phase 2: Soft Limits with Warnings (Week 3-4)
**Goal:** Guide decisions without hard blocks

**Implementation:**
1. Define correlation group thresholds:
   - HIGH correlation: > 0.7
   - MEDIUM correlation: 0.4 - 0.7
   - LOW correlation: < 0.4

2. Before entering new position, check:
   - What correlation group does this symbol belong to?
   - What's current exposure to that group?
   - Would adding this position exceed group limit?

3. If over limit:
   - Log warning (not error)
   - Still allow trade but flag for review
   - Include in daily report

**Thresholds (Proposed):**
- Max 120% per HIGH correlation group (>0.7)
- Max 180% per MEDIUM correlation group (0.4-0.7)
- No limit on LOW correlation (<0.4)

**Code Location:** New function in `src/sizing_utils.py`
```python
def check_correlation_risk(
    symbol: str,
    proposed_notional: float,
    positions: List[Position],
    correlation_matrix: np.ndarray,
) -> Dict[str, Any]:
    """
    Check if adding this position would violate correlation limits.

    Returns:
        {
            "allowed": True/False,
            "risk_level": "low"/"medium"/"high",
            "group_exposure_pct": 45.2,
            "group_limit_pct": 120.0,
            "correlated_symbols": ["AAPL", "MSFT"],
            "recommendation": "OK to proceed" or "Consider other symbols"
        }
    """
```

### Phase 3: Hard Limits (Week 5-6)
**Goal:** Enforce diversification constraints

**Implementation:**
1. Add correlation check to position sizing:
   - In `get_qty()`, before calculating quantity
   - If correlation group limit would be exceeded, reduce qty
   - Prioritize by forecast confidence/expected return

2. Add correlation bonus/penalty to position sizing:
   - Positions that diversify portfolio get size boost
   - Positions that concentrate get size reduction

**Example:**
```python
# Base quantity from existing logic
base_qty = calculate_base_qty(...)

# Correlation adjustment
portfolio_corr_with_symbol = calculate_portfolio_correlation(symbol, positions)
if portfolio_corr_with_symbol < 0.3:  # Diversifying
    adjustment = 1.2  # 20% boost
elif portfolio_corr_with_symbol > 0.7:  # Concentrating
    adjustment = 0.7  # 30% reduction
else:
    adjustment = 1.0

final_qty = base_qty * adjustment
```

### Phase 4: Optimization (Week 7+)
**Goal:** Use correlations for portfolio construction

**Advanced Features:**
1. **Correlation-aware rebalancing:**
   - When portfolio becomes too correlated, suggest uncorrelated alternatives
   - Swap correlated positions for diversifying ones

2. **Risk budgeting:**
   - Allocate risk budget across correlation groups
   - Higher return groups get more budget

3. **Pairs trading:**
   - Identify mean-reverting pairs (high correlation but temporary divergence)
   - Long-short opportunities within correlation groups

---

## Correlation Matrix Specification

### File Format
```json
{
  "timestamp": "2025-11-13T10:00:00Z",
  "lookback_days": 60,
  "symbols": ["AAPL", "MSFT", "GOOGL", ...],
  "correlation_matrix": [[1.0, 0.85, 0.72], [0.85, 1.0, 0.81], ...],
  "data_quality": {
    "AAPL": {"missing_days": 0, "data_pct": 100.0},
    "MSFT": {"missing_days": 1, "data_pct": 98.3},
    ...
  },
  "clusters": {
    "cluster_0": {
      "symbols": ["AAPL", "MSFT", "GOOGL"],
      "avg_correlation": 0.82,
      "label": "Tech Mega-Cap"
    },
    ...
  }
}
```

### Storage Options
1. **Pickle** (fast, binary): `trainingdata/correlation_matrix.pkl`
2. **JSON** (readable, portable): `trainingdata/correlation_matrix.json`
3. **HDF5** (efficient for large matrices): `trainingdata/correlation_matrix.h5`

**Recommendation:** Store both pickle (for production) and JSON (for debugging)

---

## Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Phase 1: Monitoring** | 2 weeks | Correlation matrix script, daily reports, logging |
| **Phase 2: Soft Limits** | 2 weeks | Warning system, correlation group logic |
| **Phase 3: Hard Limits** | 2 weeks | Position sizing integration, enforcement |
| **Phase 4: Optimization** | Ongoing | Advanced portfolio construction features |

**Total:** 6 weeks to full enforcement, with monitoring starting immediately

---

## Success Metrics

### Portfolio Diversification
- **Effective Number of Bets (ENB):** Target > 5 (currently unknown)
- **Average pairwise correlation:** Target < 0.5
- **Max single-group exposure:** Target < 120%

### Risk-Adjusted Performance
- **Sharpe Ratio:** Expect +10-20% improvement from better diversification
- **Max Drawdown:** Expect -10-20% reduction from avoiding correlated crashes
- **Volatility:** Expect -5-15% reduction from diversification

### Operational Metrics
- **Correlation matrix update frequency:** Daily
- **Computation time:** < 5 minutes per update
- **Data quality:** > 95% coverage for all symbols

---

## Risk Considerations

### Data Quality Issues
- **Missing data:** Use pairwise complete observations
- **Stale data:** Alert if correlation matrix is > 24 hours old
- **Insufficient history:** Require minimum 30 days of overlap

### Market Regime Changes
- **Correlation instability:** Correlations spike during crashes (all go to 1.0)
- **Solution:** Use multiple lookback windows (30d, 60d, 90d)
- **Crisis mode:** If average correlation > 0.8, increase cash reserves

### Performance Impact
- **Computation:** Matrix calculation is O(N²), but N ~300 symbols = manageable
- **Storage:** 300x300 matrix = 90K floats = ~720KB, negligible
- **Latency:** Pre-calculate daily, load from cache during trading

---

## Alternative Approaches Considered

### 1. PCA / Factor Models
- **Pros:** Identifies underlying risk factors, more interpretable
- **Cons:** More complex, requires factor labeling, harder to explain
- **Decision:** Use for Phase 4 optimization, not Phase 1-3

### 2. Fixed Sector Limits
- **Pros:** Simple, no calculations needed
- **Cons:** Misses cross-sector correlations, requires manual sector mapping
- **Decision:** Use as fallback if correlation data unavailable

### 3. VaR / CVaR Constraints
- **Pros:** Direct risk targeting
- **Cons:** Requires return distribution assumptions, harder to implement
- **Decision:** Consider for Phase 4, too complex for Phase 1

---

## Code Architecture

### New Files
```
trainingdata/
  calculate_correlation_matrix.py  # Main calculation script
  correlation_utils.py             # Helper functions
  correlation_matrix.pkl           # Latest matrix (binary)
  correlation_matrix.json          # Latest matrix (readable)
  correlation_history/             # Historical matrices
    correlation_matrix_20251113.pkl
    correlation_matrix_20251112.pkl
    ...

src/
  correlation_risk.py              # Risk checking logic
  correlation_clustering.py        # Clustering algorithms
```

### Modified Files
```
src/sizing_utils.py                # Add correlation checks to get_qty()
trade_stock_e2e.py                 # Load correlation matrix, log metrics
```

### New Dependencies
```
scipy.cluster.hierarchy  # For hierarchical clustering
sklearn.preprocessing    # For scaling/normalization
networkx                 # For graph-based analysis (optional)
```

---

## Open Questions

1. **Should we use returns or prices for correlation?**
   - **Recommendation:** Returns (stationary, better statistical properties)

2. **Should crypto and equity be in same correlation matrix?**
   - **Recommendation:** Yes, but cluster separately (crypto-crypto correlation is different)

3. **How to handle new symbols with no history?**
   - **Recommendation:** Assume average correlation with sector, flag for manual review

4. **Should we weight by recency (exponential decay)?**
   - **Recommendation:** Phase 1 use equal weights, Phase 4 add EWMA option

5. **How to handle overnight gaps for crypto (24/7)?**
   - **Recommendation:** Use hourly returns for crypto, daily for equities

---

## Next Steps

1. **Immediate:** Build `trainingdata/calculate_correlation_matrix.py`
2. **Week 1:** Run daily, validate output, build monitoring dashboard
3. **Week 2:** Implement Phase 1 (monitoring and alerts)
4. **Week 3-4:** Implement Phase 2 (soft limits)
5. **Review:** Assess impact on diversification metrics, decide on Phase 3 timeline

---

## References & Further Reading

- Ledoit-Wolf shrinkage for correlation estimation (reduces noise)
- Random Matrix Theory for correlation filtering
- "Active Portfolio Management" by Grinold & Kahn (Chapter 12: Risk)
- "Efficiently Diversified Portfolios" by Markowitz

---

## Appendix: Example Correlation Report

```
=== Daily Correlation Report ===
Date: 2025-11-13
Portfolio: LIVE

Current Positions (5):
  AAPL: $45,000 (45%)
  MSFT: $30,000 (30%)
  GOOGL: $25,000 (25%)
  BTCUSD: $20,000 (20%)
  TSLA: $15,000 (15%)

Correlation Analysis:
  Effective Number of Bets: 3.2 (target: >5) ⚠️
  Average Pairwise Correlation: 0.61 (target: <0.5) ⚠️

Correlation Groups:
  Group 1 - Tech Mega-Cap (correlation > 0.7):
    AAPL, MSFT, GOOGL
    Total Exposure: 100% (limit: 120%) ✓

  Group 2 - High Growth (correlation > 0.5):
    TSLA, BTCUSD
    Total Exposure: 35% (limit: 180%) ✓

Recommendations:
  ⚠️ Portfolio is somewhat concentrated in tech mega-caps
  ✓ Consider adding positions with correlation < 0.3:
    - TLT (Treasury bonds): corr -0.15
    - GLD (Gold): corr 0.05
    - XLE (Energy): corr 0.25
```
