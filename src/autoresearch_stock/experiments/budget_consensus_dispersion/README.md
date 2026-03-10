# Budget Consensus Dispersion

This experiment keeps the continuous budget-threshold branch, but it only
reduces broad expansion when the per-symbol budget probabilities disagree with
each other inside the same timestamp.

The key idea is:

- compute per-symbol `skip / selective / broad` probabilities,
- measure pairwise disagreement across symbols for each timestamp,
- shift only the broad mass away from expansion when disagreement is high,
- keep selective and skip behavior mostly unchanged otherwise.

Run it with:

```bash
python -m autoresearch_stock.train --frequency hourly --budget-consensus-dispersion
```

Observed hourly benchmark:

- `robust_score`: `0.333672`
- `val_loss`: `0.000322`
- `total_trade_count`: `183`

This stayed positive but regressed sharply against the stronger
`--continuous-budget-thresholds` branch. Pairwise agreement was too strict and
removed too much profitable broad expansion.
