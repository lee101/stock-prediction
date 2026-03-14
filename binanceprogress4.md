# Binance Hybrid Spot Progress (2026-03-13)

## Architecture
- Gemini 3.1 Flash Lite + Chronos2 ML forecasts
- Per-symbol LLM decisions with position context awareness
- Symbols: BTC, ETH, SOL on Binance spot
- Max hold: 6 hours, hourly decision intervals

## Prompt Variant Experiments (1-day, 2026-03-12)

| Variant | Return% | Sortino | MaxDD% | Trades | PnL |
|---------|---------|---------|--------|--------|-----|
| default | -11.124 | -15.78 | 11.449 | 9 | -$220.35 |
| **position_context** | **+0.109** | **+13.93** | **0.042** | **23** | **+$3.33** |
| fractional | -0.485 | -28.28 | 0.557 | 27 | -$7.34 |
| anonymized | -0.527 | -20.96 | 0.551 | 21 | -$8.75 |

Winner: **position_context** -- tells LLM about existing positions, P&L, hold time.

## Prompt Variant Experiments (3-day, 2026-03-12)

| Variant | Return% | Sortino | MaxDD% | Trades | PnL |
|---------|---------|---------|--------|--------|-----|
| default | +32.081 | 19.11 | 11.602 | 38 | +$808.70 |
| **position_context** | **+2.337** | **78.44** | **0.271** | **63** | **+$59.22** |

Default caught a BTC rally (Mar 9-10) but with 11.6% drawdown. Position_context: dramatically better risk-adjusted returns (Sortino 78 vs 19, MaxDD 0.27% vs 11.6%).

## Leverage Sweep (3-day, position_context, 2026-03-13)

Note: leverage >1x is hypothetical backtest leverage (borrowed notional), not spot live-parity.

| Leverage | Return% | Sortino | MaxDD% | Trades | PnL |
|----------|---------|---------|--------|--------|-----|
| 1x | +2.754 | 81.02 | 0.231 | 71 | +$63.37 |
| 2x | +5.948 | 80.98 | 0.452 | 73 | +$136.56 |
| 3x | +7.665 | 59.63 | 0.930 | 69 | +$176.62 |
| 4x | +10.643 | 53.95 | 1.059 | 63 | +$242.25 |
| **5x** | **+14.724** | **74.69** | **1.157** | **68** | **+$332.39** |

All leverage levels profitable. Sortino stays strong across the board (54-81). MaxDD scales linearly as expected: 0.23% at 1x to 1.16% at 5x -- still very low even at max leverage. 5x gives best absolute return (+14.7%) while keeping MaxDD just over 1% and Sortino at 74.7.

Sweet spot appears to be **2x** (Sortino 81, MaxDD 0.45%) or **5x** (Sortino 75, MaxDD 1.16%) depending on risk appetite. Currently deployed at 5x.

## Uncertainty Gating Experiment (3-day at 5x, 2026-03-14)

Uses Chronos2 p10/p90 CI spread to classify forecast uncertainty as LOW/MODERATE/HIGH.

| Variant | Return% | Sortino | MaxDD% | Trades | PnL |
|---------|---------|---------|--------|--------|-----|
| position_context (baseline) | +14.724 | 74.69 | 1.157 | 68 | +$332.39 |
| uncertainty_gated | +1.395 | 5.19 | 1.706 | 24 | +$38.58 |
| uncertainty_strict | +2.053 | 18.78 | 1.208 | 6 | +$44.99 |

Uncertainty gating reduces Sortino dramatically (74.7 -> 5.2/18.8). The strict variant trades very rarely (6 trades in 3 days) -- too conservative. The gated variant has *worse* MaxDD (1.7%) despite fewer trades. **Conclusion: uncertainty gating hurts. The baseline position_context prompt already handles risk well through the LLM's own reasoning.**

## Horizon Ablation Experiment (3-day at 5x, 2026-03-14)

Testing which Chronos2 forecast horizons matter.

| Variant | Return% | Sortino | MaxDD% | Trades | PnL |
|---------|---------|---------|--------|--------|-----|
| no_forecast | +2.950 | 0.00 | 0.000 | 12 | +$66.14 |
| **h1_only** | **+13.218** | **81.85** | **1.415** | **82** | **+$315.38** |
| h24_only | +9.196 | 46.27 | 0.897 | 36 | +$207.89 |
| position_context (h1+h24) | +9.448 | 61.71 | 1.382 | 61 | +$226.23 |

Key findings:
- **h1_only is the winner**: Sortino 81.85 (best), Return +13.2% (best), 82 trades
- **no_forecast barely trades** (12 trades, Sortino 0) -- forecasts are critical for giving LLM confidence to enter
- **h24_only**: lowest MaxDD (0.9%) but lower Sortino (46.3) -- longer horizon makes LLM more cautious
- **h1+h24 (baseline)**: middle ground (Sortino 61.7) -- adding h24 actually dilutes the sharper h1 signal
- **Conclusion**: h1 forecast alone gives the LLM the clearest actionable signal. h24 adds noise/caution that reduces trade frequency and Sortino.
