# Binance Transition Graph

Sequential graph-policy experiment for Binance hourly crypto rotation. It tests
rules of the form "when symbol A has rallied and symbol B has dumped, rotate
from A to B", plus regime cash-outs and optional bearish short edges.

Run a quick smoke search:

```bash
python experiments/binance_transition_graph/run_graph_experiment.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 720 \
  --eval-hours 240 \
  --trials 32 \
  --max-leverage 2 \
  --allow-short
```

Production relevance rules:

- Treat this as research only until it passes the repo gate: worst slippage
  median monthly return at least 27%, max drawdown at most 20%, decision lag at
  least 2.
- The simulator is market-on-close with explicit lag, fees, slippage, leverage,
  and borrow cost. It is useful for searching graph structure, but any promising
  result still needs the pufferlib C binary-fill evaluator before production.
