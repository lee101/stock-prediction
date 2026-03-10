# Budget Entropy Confidence

This experiment keeps the continuous budget-threshold branch, but it reduces
breadth expansion when the timestamp-level budget distribution is high-entropy.

The key idea is simple:

- compute the mean `skip / selective / broad` probabilities per timestamp,
- convert their normalized entropy into a confidence score,
- blend low-confidence hours back toward selective behavior,
- and tighten marginal breadth and threshold looseness when confidence is low.

Run it with:

```bash
python -m autoresearch_stock.train --frequency hourly --budget-entropy-confidence
```

Observed hourly benchmark:

- `robust_score`: `2.448601`
- `val_loss`: `0.000318`
- `total_trade_count`: `174`

This stayed positive but regressed against the stronger
`--continuous-budget-thresholds` branch. The confidence pull reduced breadth and
trade count, but it gave up too much profitable expansion.
