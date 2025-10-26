# StockAgent Diagnostics

This package ships an opinionated simulator plus tooling for keeping tabs on GPT generated trading plans. The project already persisted plan outcomes into `strategy_state/`; we now expose a single command that runs the test suites and prints a concise performance report.

## One-Step Test + Report

```bash
python -m scripts.run_stockagent_suite --suite stockagent
```

What this does:

- executes the `tests/prod/agents/stockagent/` test suite (pass additional `--pytest-arg` options if you want filters/verbosity)
- collects the latest state from `strategy_state/` and prints a summary with realised PnL, win rate, drawdown, top/bottom trades, and currently open exposures

> Tip: if you prefer `uv run`, make sure the toolchain is synced first:
>
> ```bash
> uv pip install -r requirements.txt
> uv run python -m scripts.run_stockagent_suite --suite stockagent
> ```

Example output:

```
=== stockagent summary ===
[stockagent] State: /path/to/repo/strategy_state (suffix _sim)
  Closed trades: 39 | Realized PnL: $-8,279.79 | Avg/trade: $-212.30 | Win rate: 10.3%
  ...
```

## Other Suites / Overrides

Multiple GPT agent stacks live in this repository and you can exercise them together:

```bash
uv run python -m scripts.run_stockagent_suite --suite stockagent --suite stockagentindependant --suite stockagent2
```

You can also point a suite at an alternate state suffix by passing `NAME:SUFFIX`:

```bash
uv run python -m scripts.run_stockagent_suite --suite stockagent:sim --suite stockagentindependant:stateless
```

If you only want the summaries and plan to run tests separately, add `--skip-tests`.

## Default Symbols & Lookback

The prompt builder now considers the full volatility set below and only pulls the most recent 30 trading days when generating requests:

```
["COUR", "GOOG", "TSLA", "NVDA", "AAPL", "U", "ADSK", "CRWD",
 "ADBE", "NET", "COIN", "META", "AMZN", "AMD", "INTC", "LCID",
 "QUBT", "BTCUSD", "ETHUSD", "UNIUSD"]
```

Update `stockagent/constants.py` if you want to experiment with a different basket.

## Reporting API

For notebooks or ad-hoc analysis, drop into Python:

```python
from stockagent.reporting import load_state_snapshot, summarize_trades, format_summary
snapshot = load_state_snapshot(state_suffix="sim")
summary = summarize_trades(snapshot=snapshot, directory=Path("strategy_state"), suffix="sim")
print(format_summary(summary, label="stockagent"))
```

The summary object exposes totals, per-symbol aggregates, and the worst/best trade lists for deeper inspection.
