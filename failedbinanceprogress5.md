# Failed Binance Experiments (2026-03-14)

## Uncertainty Gating (2026-03-14)
- uncertainty_gated: Sort=5.19, MaxDD=1.7% (vs baseline Sort=74.7)
- uncertainty_strict: Sort=18.78, only 6 trades in 3 days
- **Why failed**: Adding explicit uncertainty rules makes LLM too conservative. The baseline position_context prompt already handles risk through its own reasoning.

## Learned Meta-Selector (2026-03-10)
- Trained on raw returns, -8% to -65% on holdout
- Softmax allocation for 2-asset: mode collapse, 0 turnover
- **Why failed**: Selector operated on raw returns without Chronos2 price gating
