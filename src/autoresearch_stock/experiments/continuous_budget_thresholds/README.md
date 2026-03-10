# Continuous Budget Thresholds

This experiment keeps the existing learned timestamp budget head, but stops
reducing it to a hard `argmax` regime before the rank gate.

Instead it uses the full predicted `skip / selective / broad` probabilities to
modulate:

- candidate prefilter breadth,
- maximum survivors,
- per-timestamp score floor,
- and threshold looseness.

The goal is to smooth the handoff between narrow and broad hours so borderline
regimes are not forced into the wrong discrete bucket.

Run it with:

```bash
python -m autoresearch_stock.train --frequency hourly --continuous-budget-thresholds
```

Observed hourly benchmark:

- `robust_score`: `3.042843`
- `val_loss`: `0.000318`
- `total_trade_count`: `223`

This improved on the prior `--budget-guided-keep-count` best by replacing the
hard regime switch with probability-aware breadth and threshold control.
